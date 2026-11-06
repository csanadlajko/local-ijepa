"""Replays the batch saved by train_block_predictor (nan_batch.pt) with autograd anomaly detection.

usage: python3 scripts/replay_nan_batch.py <geometry> [nan_batch.pt]
"""
import sys

GEOM = sys.argv[1]
PATH = sys.argv[2] if len(sys.argv) > 2 else "nan_batch.pt"
sys.argv = [sys.argv[0], "--dataset", "coco", "--image_size", "256", "--num_target", "4", "--multimodal_run", "n", "--geometry", GEOM]
sys.path.insert(0, ".")

import json

import torch

from src.models.predictor import ViTPredictor
from src.models.vit import VisionTransformer
from src.utils.config_ijepa import create_finetune_model
from src.utils.logging_module import logger
from src.utils.patch_metadata import BlockProcessor, compute_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
saved = torch.load(PATH, map_location="cpu", weights_only=False)

cats = json.load(open("src/data_preprocess/coco/categories.json"))["categories"]
sup = {}
for c in cats:
    sup.setdefault(c["supercategory"], len(sup))
class_parent = [sup[c["supercategory"]] for c in cats]

kw = dict(img_size=256, patch_size=16, in_chans=3, embed_dim=256, num_heads=8, depth=6, mlp_dim=512, num_classes=80, logger=logger)
student = VisionTransformer(drop_rate=0.1, **kw).to(device)
predictor = ViTPredictor(num_patches=256, embed_dim=256, device=device, pred_dim=256, depth=6, num_heads=8,
                         num_classes=80, num_targets=4, logger=logger).to(device)
head = create_finetune_model(GEOM, 256, class_parent, 16, 2.0, 1.0, 0.5).to(device)
student.load_state_dict(saved["student"])
predictor.load_state_dict(saved["predictor"])
head.load_state_dict(saved["block_predictor"])
student.train(); predictor.train(); head.train()

images = saved["images"].to(device)
annotation = saved["annotation"]
context_masks = [m.to(device) for m in saved["context_masks"]]
target_masks = [[t.to(device) for t in tm] for tm in saved["target_masks"]]
boxes = [a["boxes"] for a in annotation]
labels, _ = BlockProcessor(16, 256, logger=logger)(boxes, target_masks, [a["string_labels"] for a in annotation], [a["labels"] for a in annotation])

print("batch", saved["batch_idx"], "| images finite:", torch.isfinite(images).all().item(),
      "| weights finite:", all(torch.isfinite(v).all().item() for k in ("student", "predictor", "block_predictor") for v in saved[k].values() if v.is_floating_point()))

if device == "cuda" and saved["rng_state"].numel() == torch.cuda.get_rng_state().numel():
    torch.cuda.set_rng_state(saved["rng_state"])
elif device == "cpu" and saved["rng_state"].numel() == torch.get_rng_state().numel():
    torch.set_rng_state(saved["rng_state"])
else:
    print("rng state from another device, dropout masks will differ")

with torch.autograd.detect_anomaly():
    tokens, _ = student(images, masks=context_masks, cls=False)
    print("student tokens finite:", torch.isfinite(tokens).all().item(), "max |x|:", tokens.abs().max().item())
    _, cls_tok = predictor(tokens, context_masks, target_masks, None, multimodal=False, return_cls_only=False, cell_mask=False, local_cls=False)
    print("block cls finite:", torch.isfinite(cls_tok).all().item(), "max |x|:", cls_tok.abs().max().item())
    flat = labels.flatten()
    logits, parts = head.losses(cls_tok, labels, class_weights=compute_weights(flat[flat != 80], 80))
    print("logits finite:", torch.isfinite(logits).all().item(), "parts:", {k: v.item() for k, v in parts.items()})
    (parts["cls"] + parts["parent"] + parts["depth"]).backward()

print("backward finished without NaN: the batch is not deterministic on this device / rng state")
