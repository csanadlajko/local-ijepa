import torch
from ..utils.masking import apply_mask
from ..utils.ema import _ema_update
import torch.nn.functional as F
from tqdm import tqdm
from ..utils.patch_metadata import compute_weights

def train_local_jepa(
        teacher_mod, 
        student_mod, 
        loader, 
        optim_student, 
        optim_predictor, 
        predictor, 
        momentum, 
        ijepa_loss,
        device,
        block_proc,
        normal_mask,
        cell_percentage=20
    ):
    
    teacher_mod.eval()
    student_mod.train()
    predictor.train()

    total_loss = 0.0
    num_batches = 0

    bar = tqdm(total=315)

    for i, (images, annotation) in enumerate(loader):
        ## annotation -> len(batch_size) list containing annotations for each image
        images = images.to(device)

        batch_bbox_list = []
        int_labels = []
        string_labels = []

        for batch in annotation:
            batch_bbox_list.append(batch["boxes"])
            int_labels.append(batch["labels"])
            string_labels.append(batch["string_labels"])

        context_masks, target_masks = normal_mask(images, batch_bbox_list)
        _, classes_string = block_proc(batch_bbox_list, target_masks, string_labels, int_labels)

        with torch.no_grad():
            ## create teacher tokens for full image
            teacher_tokens, _ = teacher_mod(images, cls=False)
            teacher_tokens = F.layer_norm(teacher_tokens, (teacher_tokens.size(-1),))
            # target tokens are padded to the longest target list in the batch!!
            # [B, N] corresponding attention mask is given
            teacher_target_tokens, _ = apply_mask(teacher_tokens, target_masks, predictor=True)

        # teacher tokens [B, N_t, D] -> ordered patch embeddings for each block
        # N_t / num_blocks patches for each block -> acquire mean pool based on that
        teacher_mean_block_tokens = teacher_target_tokens.view(teacher_target_tokens.shape[0], len(target_masks[0]), teacher_target_tokens.shape[1] // len(target_masks[0]), teacher_target_tokens.shape[2]).mean(dim=2)

        ## create context student tokens
        # student tokens are padded as well to the larges context mask index list
        student_tokens, _ = student_mod(images, masks=context_masks, cls=False)

        predicted_target_tokens, predicted_block_cls_tokens = predictor(
            student_tokens, 
            context_masks, 
            target_masks, 
            classes_string,
            multimodal=False, 
            return_cls_only=False,
            cell_mask=False,
            local_cls=False # experiment with normal ijepa otherwise its True
        )

        optim_student.zero_grad()
        optim_predictor.zero_grad()

        loss_cls = ijepa_loss(predicted_block_cls_tokens, teacher_mean_block_tokens)

        loss_curr = ijepa_loss(predicted_target_tokens, teacher_target_tokens)

        sum_loss = loss_curr + (0.7 * loss_cls)

        sum_loss.backward()
        
        optim_student.step()
        optim_predictor.step()
        
        _ema_update(teacher_mod, student_mod, momentum)
        
        total_loss += sum_loss.item()
        num_batches += 1
        bar.update(1)
        if i == 315:
            break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    bar.close()
    teacher_mod.logger.info(f"Average training loss this epoch: {avg_loss:.4f}")
    return avg_loss

def train_block_predictor(
        student_mod, 
        loader,
        optim_block_predictor,
        predictor,
        cell_predictor,
        device,
        cell_mask,
        block_processor,
        loss_fn,
        normal_mask,
        block_predictor,
        optim_student,
        optim_main_pred,
        cell_percentage=20
    ):
    block_predictor.train()
    student_mod.train()
    predictor.train()

    total_loss = 0.0
    num_batches = 0
    running_acc = 0.0

    bar = tqdm(total=315)

    epoch_topk_map = {
        "top1": 0,
        "top2": 0,
        "top3": 0,
        "top4": 0,
        "top5": 0
    }

    for i, (images, annotation) in enumerate(loader):
        ## annotation -> len(batch_size) list containing annotations for each image
        images = images.to(device)
        context_masks, target_masks = normal_mask(images)

        batch_bbox_list = []
        int_labels = []
        string_labels = []

        for batch in annotation:
            batch_bbox_list.append(batch["boxes"])
            int_labels.append(batch["labels"])
            string_labels.append(batch["string_labels"])

        context_masks, target_masks = normal_mask(images, batch_bbox_list)
        tens_int_classes, _ = block_processor(batch_bbox_list, target_masks, string_labels, int_labels)

        student_tokens, _ = student_mod(images, masks=context_masks, cls=False)

        _, block_cls_tokens = predictor(
            student_tokens, 
            context_masks, 
            target_masks, 
            None,
            multimodal=False, 
            return_cls_only=False,
            cell_mask=False,
            local_cls=False
        )

        ## average loss of all predicted target values
        # target instance mask is list[list[torch.Tensor]]

        pred_classes = block_predictor(
            block_cls_tokens
        )

        # transform into [B*target_blocks, num_classes]
        pred_classes_flat = pred_classes.view(-1, 81)

        # transform into [B*target_blocks]
        tens_int_flat = tens_int_classes.view(-1)
        weights = compute_weights(tens_int_flat, num_classes=80)
        crit = torch.nn.CrossEntropyLoss(weight=weights,ignore_index=80).to(pred_classes.device)
        loss_all = crit(pred_classes_flat, tens_int_flat)

        optim_block_predictor.zero_grad()
        optim_main_pred.zero_grad()
        optim_student.zero_grad()

        loss_all.backward()

        optim_block_predictor.step()
        optim_main_pred.step()
        optim_student.step()
        pred_labels = pred_classes_flat.argmax(dim=1)
        correct = (pred_labels == tens_int_flat)
        curr_acc = correct.sum().item() / correct.numel()
        if (i+1) % 100 == 0:
            for k in range(1, 6):
                topk_preds = pred_classes_flat.topk(k, dim=1).indices
                tens_int_flat_exp = tens_int_flat.view(-1, 1)
                correct_topk = (topk_preds == tens_int_flat_exp).any(dim=1)
                topk_acc = correct_topk.float().mean().item()
                student_mod.logger.info(f"top k acc at k={k} is {topk_acc*100:.2f}")
                epoch_topk_map[f"top{k}"] += topk_acc*100
        total_loss += loss_all.item()
        num_batches += 1
        running_acc += curr_acc
        bar.update(1)
        if i == 315:
            break

    for key in epoch_topk_map.keys():
        epoch_topk_map[key] = epoch_topk_map[key] / num_batches
    avg_loss = total_loss / num_batches
    avg_acc = (running_acc / num_batches)*100
    student_mod.logger.info(f"=== running accuracy this cell prediction epoch: {avg_acc:.2f}% ===")
    student_mod.logger.info(f"=== avg loss this cell prediction epoch: {avg_loss:.4f} ===")
    bar.close()
    return avg_loss, epoch_topk_map

def eval_block_predictor(
        student_model,
        predictor,
        test_loader,
        device,
        normal_mask,
        block_processor,
        block_predictor
    ):
    student_model.eval()
    predictor.eval()
    block_predictor.eval()

    num_batches = 0

    bar = tqdm(total=len(test_loader))
    running_acc_map = {
        "top1": 0,
        "top2": 0,
        "top3": 0,
        "top4": 0,
        "top5": 0
    }

    for i, (images, annotation) in enumerate(test_loader):
        ## annotation -> len(batch_size) list containing annotations for each image
        images = images.to(device)
        context_masks, target_masks = normal_mask(images)

        batch_bbox_list = []
        int_labels = []
        string_labels = []

        for batch in annotation:
            batch_bbox_list.append(batch["boxes"])
            int_labels.append(batch["labels"])
            string_labels.append(batch["string_labels"])

        context_masks, target_masks = normal_mask(images, batch_bbox_list)
        tens_int_classes, _ = block_processor(batch_bbox_list, target_masks, string_labels, int_labels)

        with torch.no_grad():

            student_tokens, _ = student_model(images, masks=context_masks, cls=False)

            _, block_cls_tokens = predictor(
                student_tokens, 
                context_masks, 
                target_masks, 
                None,
                multimodal=False, 
                return_cls_only=False,
                cell_mask=False,
                local_cls=False
            )

            pred_classes = block_predictor(
                block_cls_tokens
            )

        # transform into [B*target_blocks, num_classes]
        pred_classes_flat = pred_classes.view(-1, 81)

        # transform into [B*target_blocks]
        tens_int_flat = tens_int_classes.view(-1)
        for k in range(1, 6):
            topk_preds = pred_classes_flat.topk(k, dim=1).indices
            tens_int_flat_exp = tens_int_flat.view(-1, 1)
            correct_topk = (topk_preds == tens_int_flat_exp).any(dim=1)
            topk_acc = correct_topk.float().mean().item()
            topk_key = f"top{k}"
            running_acc_map[topk_key] += topk_acc

        num_batches += 1
        if (i+1) % 200 == 0:
            for key in running_acc_map.keys():
                predictor.logger.info(f"running accuracy for {key} is: {running_acc_map[key] / num_batches * 100}%")

        bar.update(1)
    for key in running_acc_map.keys():
        running_acc_map[key] = (running_acc_map[key] / num_batches) * 100
    
    return running_acc_map