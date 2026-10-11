"""Extracts frozen block features for the linear probe and the frozen-backbone head experiments.

For every non-background target block two features are saved:
  pred_cls      the predictor's block CLS token, computed from the context only (the block is masked)
  teacher_mean  the mean of the teacher's patch tokens inside the block, computed from the full image

Without checkpoints the models stay randomly initialised: this is the untrained-backbone control.
The train/test split and the masks depend only on --seed, so files extracted with the same seed
contain the same blocks and can be compared directly.

Use the same arguments as in `runwith`, plus:
    python3 tools/extract_features.py <runwith arguments> --out feats_jepa20.pt \\
        --student_ckpt trained_student_jepa_no_mm_<run>_20ep.pth \\
        --teacher_ckpt teacher_model_jepa_no_mm_<run>_20ep.pth \\
        --predictor_ckpt trained_predictor_jepa_no_mm_<run>_20ep.pth
    python3 tools/extract_features.py <runwith arguments> --out feats_random.pt
"""
import argparse
import os

from common import parse_own_args

own = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
own.add_argument("--out", required=True, help="output .pt file")
own.add_argument("--student_ckpt", default="")
own.add_argument("--teacher_ckpt", default="")
own.add_argument("--predictor_ckpt", default="")
own.add_argument("--train_batches", type=int, default=1000, help="number of training batches to extract (default: 1000)")
own.add_argument("--test_batches", type=int, default=500, help="number of test batches to extract (default: 500)")
own.add_argument("--seed", type=int, default=0)
own_args = parse_own_args(own, path_keys=("out", "student_ckpt", "teacher_ckpt", "predictor_ckpt"))

import torch

from common import (BACKGROUND, block_targets, build_backbone, coco_loaders, load_checkpoint, predictor_block_cls,
                    seed_everything, teacher_block_mean)
from src.parser.parser import parse_jepa_args
from src.utils.logging_module import logger
from src.utils.masking import Mask
from src.utils.patch_metadata import BlockProcessor

args = parse_jepa_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

seed_everything(own_args.seed)
student, teacher, predictor = build_backbone(args, device)
loaded = []
for model, path in ((student, own_args.student_ckpt), (teacher, own_args.teacher_ckpt), (predictor, own_args.predictor_ckpt)):
    if path:
        load_checkpoint(model, path, device)
        loaded.append(os.path.basename(path))
print("checkpoints:", ", ".join(loaded) if loaded else "none, random initialisation")

# reseeding after the models makes the split and the masks identical with and without checkpoints
seed_everything(own_args.seed)
train_loader, test_loader, class_parent = coco_loaders(args, own_args.seed)
mask = Mask(device=device)
block_proc = BlockProcessor(args.patch_size, args.image_size, logger=logger)
for model in (student, teacher, predictor):
    model.eval()


def extract(loader, max_batches, split):
    feats = {"pred_cls": [], "teacher_mean": [], "labels": []}
    with torch.no_grad():
        for i, (images, annotation) in enumerate(loader):
            if i == max_batches:
                break
            images = images.to(device)
            context_masks, target_masks, labels = block_targets(images, annotation, mask, block_proc)
            valid = labels.flatten() != BACKGROUND
            pred_cls = predictor_block_cls(student, predictor, images, context_masks, target_masks).flatten(0, 1)
            teacher_mean = teacher_block_mean(teacher, images, target_masks).flatten(0, 1)
            feats["pred_cls"].append(pred_cls[valid].half().cpu())
            feats["teacher_mean"].append(teacher_mean[valid].half().cpu())
            feats["labels"].append(labels.flatten()[valid].cpu())
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1} batches")
    return {key: torch.cat(value) for key, value in feats.items()}


train = extract(train_loader, own_args.train_batches, "train")
test = extract(test_loader, own_args.test_batches, "test")
torch.save({
    "train": train,
    "test": test,
    "class_parent": [int(c) for c in class_parent],
    "checkpoints": loaded,
    "seed": own_args.seed,
}, own_args.out)
print(f"saved {len(train['labels'])} train and {len(test['labels'])} test blocks to {own_args.out}")
