"""Evaluates a finetuned run from its saved checkpoints.

Prints the accuracy next to the class-frequency baseline, the balanced accuracy, what the head
actually predicts, the per-class accuracy of the most frequent classes, and whether the block
features differ between blocks at all.

Use the same arguments as in `runwith` (geometry and finetune_embed_dim as in the run), plus:
    python3 tools/diagnose_finetuned.py <runwith arguments> \\
        --student_ckpt trained_student_cls_no_mm_<run>_20ep.pth \\
        --predictor_ckpt trained_predictor_cls_no_mm_<run>_20ep.pth \\
        --head_ckpt trained_block_pred_no_mm_<run>_20ep.pth
"""
import argparse

from common import parse_own_args

own = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
own.add_argument("--student_ckpt", required=True)
own.add_argument("--predictor_ckpt", required=True)
own.add_argument("--head_ckpt", required=True)
own.add_argument("--max_batches", type=int, default=300, help="number of test batches to evaluate (default: 300)")
own.add_argument("--seed", type=int, default=0)
own_args = parse_own_args(own, path_keys=("student_ckpt", "predictor_ckpt", "head_ckpt"))

import torch

from common import (BACKGROUND, block_targets, build_backbone, category_names, coco_loaders, evaluate_logits,
                    format_row, load_checkpoint, predictor_block_cls, prior_logits, seed_everything)
from src.parser.parser import parse_jepa_args
from src.utils.config_ijepa import create_finetune_model
from src.utils.logging_module import logger
from src.utils.masking import Mask
from src.utils.patch_metadata import BlockProcessor

args = parse_jepa_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

student, _, predictor = build_backbone(args, device)
load_checkpoint(student, own_args.student_ckpt, device)
load_checkpoint(predictor, own_args.predictor_ckpt, device)

seed_everything(own_args.seed)
_, test_loader, class_parent = coco_loaders(args, own_args.seed)
head = create_finetune_model(
    geometry=args.geometry,
    jepa_embed_dim=args.embed_dim,
    class_parent=class_parent,
    proj_dim=args.finetune_embed_dim,
    r_max=args.r_max,
    tau=args.tau,
    margin=args.margin
).to(device)
load_checkpoint(head, own_args.head_ckpt, device)

mask = Mask(device=device)
block_proc = BlockProcessor(args.patch_size, args.image_size, logger=logger)
for model in (student, predictor, head):
    model.eval()

all_logits, all_labels, all_feats = [], [], []
with torch.no_grad():
    for i, (images, annotation) in enumerate(test_loader):
        if i == own_args.max_batches:
            break
        images = images.to(device)
        context_masks, target_masks, labels = block_targets(images, annotation, mask, block_proc)
        block_cls = predictor_block_cls(student, predictor, images, context_masks, target_masks)
        valid = labels.flatten() != BACKGROUND
        all_logits.append(head(block_cls).flatten(0, 1)[valid].cpu())
        all_labels.append(labels.flatten()[valid].cpu())
        all_feats.append(block_cls.flatten(0, 1)[valid].cpu())

logits, labels, feats = torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_feats)
names = category_names()

print(f"\nevaluated non-background blocks: {len(labels)}  (geometry: {args.geometry}, head dim: {args.finetune_embed_dim})\n")
print(format_row("class-frequency baseline (always most frequent)", evaluate_logits(prior_logits(labels, len(labels)), labels, class_parent), names))
print(format_row(f"finetuned {args.geometry} head", evaluate_logits(logits, labels, class_parent), names))

pred = logits.argmax(dim=1)
total = torch.bincount(labels, minlength=80)
correct = torch.bincount(labels[pred == labels], minlength=80)
predicted = torch.bincount(pred, minlength=80)
print("\nmost frequent classes:")
for c in total.argsort(descending=True)[:10].tolist():
    print(f"  {names[c]:<15} share {total[c] / len(labels) * 100:5.1f}%   accuracy {correct[c] / max(total[c], 1) * 100:5.1f}%   predicted for {predicted[c] / len(labels) * 100:5.1f}% of blocks")

print(f"\nblock CLS features: mean per-dimension std across blocks {feats.float().std(dim=0).mean():.3f} "
      f"(0: every block gets the same vector, around 1: as varied as independent layer-normed vectors)")
