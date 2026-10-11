"""Shared helpers for the diagnostic and frozen-feature scripts in tools/."""
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CLASSES = 80
BACKGROUND = 80


def parse_own_args(parser, path_keys=()):
    # the repo modules call parse_jepa_args() on import, so only the repo arguments may stay in sys.argv
    own_args, rest = parser.parse_known_args()
    for key in path_keys:
        value = getattr(own_args, key)
        if isinstance(value, list):
            setattr(own_args, key, [os.path.abspath(v) for v in value])
        elif value:
            setattr(own_args, key, os.path.abspath(value))
    sys.argv = [sys.argv[0]] + rest
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    # categories.json is opened with a path relative to the repo root
    os.chdir(REPO_ROOT)
    return own_args


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def category_names():
    with open(os.path.join(REPO_ROOT, "src/data_preprocess/coco/categories.json")) as f:
        return [c["name"] for c in json.load(f)["categories"]]


def build_backbone(args, device):
    """Student, teacher and predictor built exactly as in local_jepa.py (without the text encoder)."""
    from src.models.vit import VisionTransformer
    from src.models.predictor import ViTPredictor
    from src.utils.config_ijepa import init_weights
    from src.utils.logging_module import logger

    def vit(drop_rate):
        return VisionTransformer(
            img_size=args.image_size,
            patch_size=args.patch_size,
            in_chans=args.channels,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            depth=args.depth,
            mlp_dim=args.mlp_dim,
            drop_rate=drop_rate,
            num_classes=args.num_classes,
            logger=logger
        ).to(device)

    student = vit(args.student_dropout)
    teacher = vit(args.teacher_dropout)
    predictor = ViTPredictor(
        num_patches=student.patch_embed.num_patches,
        embed_dim=args.embed_dim,
        device=device,
        pred_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        num_targets=args.num_target,
        logger=logger
    ).to(device)

    student.apply(init_weights)
    predictor.apply(init_weights)
    teacher.load_state_dict(student.state_dict())
    return student, teacher, predictor


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    # checkpoints saved while the text encoder was still a submodule of the predictor
    for key in [k for k in state if k.startswith("text_encoder.")]:
        del state[key]
    model.load_state_dict(state)


def coco_loaders(args, seed):
    from src.data_preprocess.dataloader import load_coco_dataset

    # random_split uses the global generator, seeding it fixes the train/test split
    torch.manual_seed(seed)
    train_loader, test_loader, _, class_parent = load_coco_dataset(
        args.coco_train_folder, args.coco_train_annotation, args.reverse_transform
    )
    return train_loader, test_loader, class_parent


def block_targets(images, annotation, mask, block_proc):
    bboxes = [a["boxes"] for a in annotation]
    context_masks, target_masks = mask(images, bboxes)
    labels, _ = block_proc(bboxes, target_masks, [a["string_labels"] for a in annotation], [a["labels"] for a in annotation])
    return context_masks, target_masks, labels


def predictor_block_cls(student, predictor, images, context_masks, target_masks):
    """[B, T, D] block CLS tokens, computed from the context only, as in finetuning."""
    student_tokens, _ = student(images, masks=context_masks, cls=False)
    _, block_cls = predictor(
        student_tokens,
        context_masks,
        target_masks,
        None,
        multimodal=False,
        return_cls_only=False,
        cell_mask=False,
        local_cls=False
    )
    return block_cls


def teacher_block_mean(teacher, images, target_masks):
    """[B, T, D] mean of the layer-normed teacher patch tokens inside each block, from the full image."""
    tokens, _ = teacher(images, cls=False)
    tokens = F.layer_norm(tokens, (tokens.size(-1),))
    return torch.stack([
        torch.stack([tokens[i, idx.reshape(-1)].mean(dim=0) for idx in blocks])
        for i, blocks in enumerate(target_masks)
    ])


def evaluate_logits(logits, labels, class_parent):
    """Metrics over non-background blocks. logits: [N, 80], labels: [N]."""
    top5 = logits.topk(5, dim=1).indices
    hits = top5 == labels[:, None]
    pred = top5[:, 0]
    total = torch.bincount(labels, minlength=N_CLASSES)
    correct = torch.bincount(labels[pred == labels], minlength=N_CLASSES)
    seen = total > 0
    parent = torch.as_tensor(class_parent, device=labels.device)
    hist = torch.bincount(pred, minlength=N_CLASSES)
    return {
        "top1": hits[:, :1].any(dim=1).float().mean().item() * 100,
        "top5": hits.any(dim=1).float().mean().item() * 100,
        "balanced": (correct[seen].float() / total[seen]).mean().item() * 100,
        "super": (parent[pred] == parent[labels]).float().mean().item() * 100,
        "distinct": int((hist > 0).sum()),
        "top_pred": int(hist.argmax()),
        "top_pred_share": hist.max().item() / max(hist.sum().item(), 1) * 100,
    }


def prior_logits(train_labels, n_test):
    """Always the same ranking: the classes ordered by their training frequency."""
    freq = torch.bincount(train_labels, minlength=N_CLASSES).float()
    return (freq + 1e-3).log().expand(n_test, N_CLASSES)


def format_row(name, m, names=None):
    top_pred = names[m["top_pred"]] if names else str(m["top_pred"])
    return (
        f"{name:<46} top1 {m['top1']:5.1f}  top5 {m['top5']:5.1f}  balanced {m['balanced']:5.1f}  "
        f"super {m['super']:5.1f}  distinct {m['distinct']:2d}  most predicted: {top_pred} ({m['top_pred_share']:.0f}%)"
    )
