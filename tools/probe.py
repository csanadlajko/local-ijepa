"""Trains classification heads on frozen block features saved by extract_features.py.

On the same blocks it compares the class-frequency baseline, a linear probe and the euclidean,
hyperbolic and spheric prototype heads, with and without class-balanced loss. Runs in minutes,
also on CPU, so it can be repeated for several head dimensions and seeds.

    python3 tools/probe.py --features feats_jepa20.pt feats_random.pt
    python3 tools/probe.py --features feats_jepa20.pt --feature_types pred_cls --heads euclidean hyperbolic spheric --dims 4 16
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import N_CLASSES, category_names, evaluate_logits, format_row, prior_logits
from src.utils.config_ijepa import create_finetune_model, init_weights, param_groups

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--features", nargs="+", required=True, help=".pt files from extract_features.py")
parser.add_argument("--feature_types", nargs="+", default=["pred_cls", "teacher_mean"], choices=["pred_cls", "teacher_mean"])
parser.add_argument("--heads", nargs="+", default=["linear", "euclidean", "hyperbolic", "spheric"],
                    choices=["linear", "euclidean", "hyperbolic", "spheric"])
parser.add_argument("--dims", nargs="+", type=int, default=[16], help="head dimensions of the prototype heads (default: 16)")
parser.add_argument("--balanced", default="both", choices=["y", "n", "both"], help="class-balanced cross-entropy (default: both)")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lam_parent", type=float, default=1.0)
parser.add_argument("--lam_depth", type=float, default=1.0)
parser.add_argument("--r_max", type=float, default=2.0)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--margin", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
names = category_names()


def balanced_weights(labels):
    counts = torch.bincount(labels, minlength=N_CLASSES).float()
    present = counts > 0
    weights = torch.zeros(N_CLASSES, device=labels.device)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights


def train_head(head, dim, x_train, y_train, class_parent, weights):
    torch.manual_seed(args.seed)
    if head == "linear":
        model = torch.nn.Linear(x_train.shape[1], N_CLASSES)
        init_weights(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        model = create_finetune_model(
            geometry=head,
            jepa_embed_dim=x_train.shape[1],
            class_parent=class_parent,
            proj_dim=dim,
            r_max=args.r_max,
            tau=args.tau,
            margin=args.margin
        )
        model.apply(init_weights)
        optimizer = torch.optim.AdamW(param_groups(model, 0.05, model.no_weight_decay()), lr=args.lr)
    model.to(device)

    for _ in range(args.epochs):
        perm = torch.randperm(len(x_train), device=device)
        for start in range(0, len(x_train), args.batch_size):
            idx = perm[start:start + args.batch_size]
            x, y = x_train[idx], y_train[idx]
            if head == "linear":
                loss = F.cross_entropy(model(x), y, weight=weights)
            else:
                logits, parts = model.losses(x[:, None], y[:, None])
                loss = (F.cross_entropy(logits.flatten(0, 1), y, weight=weights)
                        + args.lam_parent * parts["parent"] + args.lam_depth * parts["depth"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def predict(model, head, x):
    with torch.no_grad():
        chunks = [model(c) if head == "linear" else model(c[:, None]).squeeze(1) for c in x.split(8192)]
    return torch.cat(chunks)


def geometry_stats(model, x):
    with torch.no_grad():
        z = model.to_manifold(model.clip(model.embed_proj(x)))
        p = model.to_manifold(model.clip(model.class_nodes))
        s = model.to_manifold(model.clip(model.category_nodes))
        violations = (model.radius(p) <= model.radius(s)[model.class_parent]).float().mean().item() * 100
        return f"   class not deeper than its parent: {violations:4.0f}%   mean block radius: {model.radius(z).mean().item():.2f}"


balanced_options = {"y": [True], "n": [False], "both": [False, True]}[args.balanced]

for path in args.features:
    data = torch.load(path, map_location="cpu")
    class_parent = data["class_parent"]
    y_train = data["train"]["labels"].to(device)
    y_test = data["test"]["labels"].to(device)
    source = ", ".join(data["checkpoints"]) if data["checkpoints"] else "random initialisation"
    print(f"\n=== {os.path.basename(path)}  ({source})  train blocks: {len(y_train)}, test blocks: {len(y_test)}")
    print(format_row("class-frequency baseline", evaluate_logits(prior_logits(y_train, len(y_test)).to(device), y_test, class_parent), names))

    for feature_type in args.feature_types:
        x_train = data["train"][feature_type].float().to(device)
        x_test = data["test"][feature_type].float().to(device)
        mean, std = x_train.mean(dim=0), x_train.std(dim=0).clamp_min(1e-6)
        x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        print(f"--- {feature_type}")

        for head in args.heads:
            for dim in ([None] if head == "linear" else args.dims):
                for balanced in balanced_options:
                    weights = balanced_weights(y_train) if balanced else None
                    model = train_head(head, dim, x_train, y_train, class_parent, weights)
                    metrics = evaluate_logits(predict(model, head, x_test), y_test, class_parent)
                    name = f"{head}{'' if dim is None else f' {dim}D'}{' balanced' if balanced else ''}"
                    extra = "" if head == "linear" else geometry_stats(model, x_test)
                    print(format_row(name, metrics, names) + extra)
