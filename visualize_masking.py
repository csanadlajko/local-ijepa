
### DEPRACATED SCRIPT, WILL BE UPDATE IN FUTURE VERSIONS ###

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_preprocess.dataloader import load_coco_dataset
from src.models.predictor import BlockTypePredictor, ViTPredictor
from src.models.vit import VisionTransformer
from src.parser.parser import parse_jepa_args
from src.utils.masking import Mask
from src.utils.patch_metadata import BlockProcessor

args = parse_jepa_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, _ = load_coco_dataset(args.coco_train_folder, args.coco_train_annotation, "y")

def visualize_masks(images, context_masks, target_masks, bbox_list=None, string_labels=None, num_samples=4):
    batch_size = min(images.shape[0], num_samples)
    patch_size = args.patch_size
    img_size = args.image_size
    num_patches_per_side = img_size // patch_size
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        img = images[i].cpu()
        img = img.permute(1, 2, 0).numpy()
        img = np.clip((img + 1) / 2, 0, 1) 
        
        context_mask_2d = torch.zeros(num_patches_per_side, num_patches_per_side)
        target_mask_2d = torch.zeros(num_patches_per_side, num_patches_per_side)
        
        if isinstance(context_masks[i], torch.Tensor) and context_masks[i].numel() > 0:
            ctx_indices = context_masks[i].cpu()
            for idx in ctx_indices:
                row = idx // num_patches_per_side
                col = idx % num_patches_per_side
                if 0 <= row < num_patches_per_side and 0 <= col < num_patches_per_side:
                    context_mask_2d[row, col] = 1
        
        if isinstance(target_masks[i], list):
            for block_idx in target_masks[i]:
                if isinstance(block_idx, torch.Tensor) and block_idx.numel() > 0:
                    for idx in block_idx.cpu():
                        row = idx // num_patches_per_side
                        col = idx % num_patches_per_side
                        if 0 <= row < num_patches_per_side and 0 <= col < num_patches_per_side:
                            target_mask_2d[row, col] = 1
        
        context_mask_2d_np = context_mask_2d.numpy()
        target_mask_2d_np = target_mask_2d.numpy()
        
        context_mask_img_2d = np.kron(context_mask_2d_np, np.ones((patch_size, patch_size)))
        target_mask_img_2d = np.kron(target_mask_2d_np, np.ones((patch_size, patch_size)))
        
        if context_mask_img_2d.shape[0] != img_size or context_mask_img_2d.shape[1] != img_size:
            h_ratio = img_size / context_mask_img_2d.shape[0]
            w_ratio = img_size / context_mask_img_2d.shape[1]
            context_mask_img_2d = np.repeat(np.repeat(context_mask_img_2d, int(h_ratio), axis=0), int(w_ratio), axis=1)
            target_mask_img_2d = np.repeat(np.repeat(target_mask_img_2d, int(h_ratio), axis=0), int(w_ratio), axis=1)
            context_mask_img_2d = context_mask_img_2d[:img_size, :img_size]
            target_mask_img_2d = target_mask_img_2d[:img_size, :img_size]
        
        axes[i, 0].imshow(img)
        if bbox_list and i < len(bbox_list) and string_labels and i < len(string_labels):
            for j, box in enumerate(bbox_list[i]):
                x, y, w, h = box
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axes[i, 0].add_patch(rect)
                if j < len(string_labels[i]):
                    axes[i, 0].text(x, y-5, string_labels[i][j], color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        axes[i, 0].set_title(f'Original Image {i+1} with BBoxes')
        axes[i, 0].axis('off')
        
        context_mask_bool = context_mask_img_2d == 1
        context_overlay = np.zeros_like(img)
        context_overlay[:, :, 1] = context_mask_bool.astype(float)
        
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(context_overlay, alpha=0.5)
        axes[i, 1].set_title(f'Context Mask (green) - Image {i+1}')
        axes[i, 1].axis('off')
        
        target_mask_bool = target_mask_img_2d == 1
        target_overlay = np.zeros_like(img)
        target_overlay[:, :, 0] = target_mask_bool.astype(float)
        
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(target_overlay, alpha=0.5)
        axes[i, 2].set_title(f'Target Masks (red) - Image {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

text_encoder = AutoModelForCausalLM.from_pretrained(args.sentence_encoder).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.sentence_encoder)

teacher_model = VisionTransformer(
    img_size=args.image_size,
    patch_size=args.patch_size,
    in_chans=args.channels,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    depth=args.depth,
    mlp_dim=args.mlp_dim,
    drop_rate=args.teacher_dropout,
    num_classes=args.num_classes
).to(device)

predictor = ViTPredictor(
    num_patches=teacher_model.patch_embed.num_patches,
    embed_dim=args.embed_dim,
    device=device,
    pred_dim=args.embed_dim,
    depth=args.depth,
    num_heads=args.num_heads,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    num_classes=args.num_classes,
    num_targets=args.num_target
).to(device)

student_model = VisionTransformer(
    img_size=args.image_size,
    patch_size=args.patch_size,
    in_chans=args.channels,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    depth=args.depth,
    mlp_dim=args.mlp_dim,
    drop_rate=args.student_dropout,
    num_classes=args.num_classes
).to(device)

predictor = ViTPredictor(
        num_patches=teacher_model.patch_embed.num_patches,
        embed_dim=args.embed_dim,
        device=device,
        pred_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        num_classes=args.num_classes,
        num_targets=args.num_target
    ).to(device)

block_proc = BlockProcessor(args.patch_size, args.image_size)

block_predictor = BlockTypePredictor(embed_dim=args.embed_dim, num_classes=args.num_block_categories).to(device)

teacher_model.load_state_dict(torch.load("results/trained_models/multimodal/fine-tuned/teacher/teacher_model_cls_2026-04-15T081858Z_40ep.pth", weights_only=True))
predictor.load_state_dict(torch.load("results/trained_models/multimodal/fine-tuned/predictor/trained_predictor_cls_2026-04-15T081858Z_40ep.pth", weights_only=True))
student_model.load_state_dict(torch.load("results/trained_models/multimodal/fine-tuned/student/trained_student_cls_2026-04-15T081858Z_40ep.pth", weights_only=True))
block_predictor.load_state_dict(torch.load("results/trained_models/multimodal/block-classifier/trained_block_pred_2026-04-15T081858Z_40ep.pth", weights_only=True))

with open("src/data_preprocess/coco/categories.json") as f:
    categories = json.load(f)

if __name__ == "__main__":
    mask = Mask(device=device)
    proc = BlockProcessor(16, 512)
    
    reverse_map = {}
    for key, value in proc.categ_map.items():
        reverse_map[str(value)] = int(key) + 1

    for images, labels in train_loader:
        images = images.cuda() if torch.cuda.is_available() else images


        batch_bbox_list = []
        int_labels = []
        string_labels = []

        for batch in labels:
            batch_bbox_list.append(batch["boxes"])
            int_labels.append(batch["labels"])
            string_labels.append(batch["string_labels"])
        
        context_masks, target_masks = mask(images, batch_bbox_list)

        gt, _ = proc(batch_bbox_list, target_masks, string_labels, int_labels)
        with torch.no_grad():
            student_tokens, _ = student_model(images, masks=context_masks)

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
        topk_preds = pred_classes_flat.topk(5, dim=1).indices

        topk_list = topk_preds.tolist()
        string_predictions = []
        temp_arr = []
        for block_idx in range(1, (16*4)+1):
            if (block_idx % 5) == 0:
                string_predictions.append(temp_arr)
                temp_arr = []
            else:
                gt_ids = [reverse_map[str(pidx)] for pidx in topk_list[block_idx-1]]
                gt_strings = []
                for id in gt_ids:
                    for categ in categories["categories"]:
                        if categ["id"] == id:
                            gt_strings.append(categ["name"])
                            break
                temp_arr.append(gt_strings)

        for i, pred in enumerate(string_predictions):
            print(f"block predictions for image {i+1}")
            print(pred)
        print(f"Batch shape: {images.shape}")
        print(f"Number of context masks: {len(context_masks)}")
        print(f"Number of target mask sets: {len(target_masks)}")
        print(f"Target blocks per image: {[len(tm) if isinstance(tm, list) else 1 for tm in target_masks]}")
        
        for i, tm in enumerate(target_masks):
            if isinstance(tm, list):
                total_tokens = sum(block.numel() if isinstance(block, torch.Tensor) else 0 for block in tm)
                print(f"Image {i}: {len(tm)} target blocks, {total_tokens} total target tokens")
        
        fig = visualize_masks(images.cpu(), context_masks, target_masks, batch_bbox_list, string_labels)
        plt.savefig('mask_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to mask_visualization.png")
        plt.show()
        
        break

