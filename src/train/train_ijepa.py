import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..utils.ema import _ema_update
from ..utils.masking import apply_mask

run_identifier: str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def train(teacher_mod, 
          student_mod,
          loader, 
          optim_student, 
          optim_predictor, 
          predictor, 
          momentum, 
          ijepa_loss,
          device,
          mask,
          multimodal=True,
          debug="n"):
    teacher_mod.eval()
    student_mod.train()
    predictor.train()

    total_loss = 0.0
    num_batches = 0

    bar = tqdm(total=len(loader))

    for (images, labels) in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        context_masks, target_masks = mask(images) # only indices -> exclude cls
        
        with torch.no_grad():
            teacher_tokens, _ = teacher_mod(images)
            teacher_tokens = F.layer_norm(teacher_tokens, (teacher_tokens.size(-1),))
            teacher_target_tokens, _ = apply_mask(teacher_tokens, target_masks)

        student_tokens, _ = student_mod(images, masks=context_masks)

        # if debug == "y":
        #     make_dot(student_tokens, params=dict(student_mod.named_parameters())).render(filename="model_vis", directory="results", format="png")

        predicted_target_tokens = predictor(student_tokens, context_masks, target_masks, labels, multimodal, return_cls_only=False)
        
        optim_student.zero_grad()
        optim_predictor.zero_grad()
        
        loss_curr = ijepa_loss(predicted_target_tokens, teacher_target_tokens)
            
        loss_curr.backward()
        
        optim_student.step()
        optim_predictor.step()
        
        _ema_update(teacher_mod, student_mod, momentum)
        
        total_loss += loss_curr.item() 
        num_batches += 1

        bar.update(1)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    teacher_mod.logger.info(f"Average training loss: {avg_loss:.4f}")

    bar.close()

    return avg_loss

def train_cls(student_model,
              train_dataset,
              predictor,
              optim_cls,
              cls_loss,
              device,
              mask,
              multimodal=False):
    student_model.eval() # freeze trained student model
    predictor.train()
    
    total_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0

    for name, param in student_model.named_parameters():
        if "cls_fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    bar = tqdm(total=len(train_dataset))
    
    for batch_idx, (images, labels) in enumerate(train_dataset):
        images = images.to(device)
        labels = labels.to(device)

        ctx_mask, trgt_masks = mask(images)
        
        # student mask embeddings from trained model
        student_enc, _ = student_model(images, masks=ctx_mask)

        # cls tokens for the predicted full image -> not training cls from context embeddings
        pred_classes = predictor(student_enc, ctx_mask, trgt_masks, labels, multimodal, return_cls_only=True)

        optim_cls.zero_grad()

        loss = cls_loss(pred_classes, labels)
        loss.backward()
        optim_cls.step()

        _, predicted = torch.max(pred_classes, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        total_loss += loss.item()
        num_batches += 1

        bar.update(1)

    current_acc = correct_predictions / total_predictions
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    student_model.logger.info(f"Average CLS training loss: {avg_loss:.4f}")

    bar.close()

    return avg_loss, current_acc * 100

def eval_cls(model, 
            test_dataset,
            predictor,
            device,
            mask,
            multimodal=False):
    """
    Evaluate the model using CLS token classification
    """
    model.eval()
    predictor.eval()
    total_correct = 0
    total_samples = 0

    bar = tqdm(total=len(test_dataset))
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataset):
            images = images.to(device)
            labels = labels.to(device)

            ctx_masks, target_masks = mask(images)
            
            student_embeddings, _ = model(images, masks=ctx_masks)

            pred_classes = predictor(student_embeddings, ctx_masks, target_masks, labels, multimodal, return_cls_only=True)

            _, predicted = torch.max(pred_classes, 1)
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            bar.update(1)
            
    bar.close()
    final_accuracy = total_correct / total_samples
    predictor.logger.info(f"Final CLS accuracy: {final_accuracy:.4f}")
    return final_accuracy

def show_loss_per_epoch(cls_loss_per_epoch: list[int], run_id: str, result_folder: str):
    epoch_list = range(1, len(cls_loss_per_epoch) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, cls_loss_per_epoch, label="CE loss per finetuning epochs")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.title('Loss over epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_folder}/jepa_loss_plot_{run_id}.png', dpi=500)
    plt.show()

def show_cls_data_per_epoch(accuracy_per_epoch: list[int], run_id: str, result_folder: str):
    epoch_list = range(1, len(accuracy_per_epoch) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, accuracy_per_epoch, label="Accuracy per epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy per epoch (%)')
    plt.title("CLS accuracy per epoch (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_folder}/cls_accuracy_plot_{run_id}.png', dpi=300)
    plt.show()

def show_topk_accuracy(topk_acc: list[dict[str, float]], run_id: str, result_folder: str):
    epoch_list = range(1, len(topk_acc) + 1)
    top1 = [t["top1"] for t in topk_acc]
    top2 = [t["top2"] for t in topk_acc]
    top3 = [t["top3"] for t in topk_acc]
    top4 = [t["top4"] for t in topk_acc]
    top5 = [t["top5"] for t in topk_acc]
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, top1, label="Top-1 Accuracy", color="blue")
    plt.plot(epoch_list, top2, label="Top-2 Accuracy", color="orange")
    plt.plot(epoch_list, top3, label="Top-3 Accuracy", color="green")
    plt.plot(epoch_list, top4, label="Top-4 Accuracy", color="red")
    plt.plot(epoch_list, top5, label="Top-5 Accuracy", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Top-k Accuracy")
    plt.title("Top-k accuracy per epoch (%)")
    plt.legend()
    plt.grid()
    plt.savefig(f'{result_folder}/topk_accuracy_{run_id}.png', dpi=300)
    plt.show()

def show_topk_barchart(topk_acc_dict: dict[str, float], result_folder: str, run_id: str, title: str, categories: list[str] = ["Top-1", "Top-2", "Top-3", "Top-4", "Top-5"]):
    values: list[float] = [val for val in topk_acc_dict.values()]
    plt.bar(categories, values, color=['#cfe8ff', '#9ecbff', '#6ea8ff', '#3b7dff', '#1f3fbf'], edgecolor="black")
    plt.title(title)
    plt.xlabel("Top-k accuracy categories")
    plt.ylabel("Accuracy scores (%)")
    plt.savefig(f'{result_folder}/topk_barchart_{run_id}.png', dpi=300)
    plt.show()