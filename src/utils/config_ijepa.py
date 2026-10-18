import torch
from torch import nn

from ..utils.embed_space import (
    EuclideanEmbeddingSpaceHead,
    HyperbolicEmbeddingSpaceHead,
    SphericEmbeddingSpaceHead,
)


def param_groups(model, weight_decay, no_decay=()):
    return [
        {"params": [p for n, p in model.named_parameters() if n not in no_decay and p.ndim > 1], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n in no_decay or p.ndim <= 1], "weight_decay": 0.0}
    ]

def get_model_config(student_model, predictor, learning_rate, epochs, block_predictor, finetune_lr=0.005):

    optim_student = torch.optim.AdamW(
        param_groups(student_model, 0.05),
        lr=learning_rate
    )

    optim_predictor = torch.optim.AdamW(
        param_groups(predictor, 0.05),
        lr=learning_rate
    )

    optim_cls = torch.optim.AdamW(
        param_groups(predictor, 0.05),
        lr=finetune_lr
    )

    optim_block_predictor = torch.optim.AdamW(
        param_groups(block_predictor, 0.05, block_predictor.no_weight_decay()),
        lr=finetune_lr
    )

    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_student,
        epochs
    )

    return {
        "optim_student": optim_student,
        "optim_cls": optim_cls,
        "optim_predictor": optim_predictor,
        "optim_block_predictor": optim_block_predictor,
        "ijepa_loss": nn.MSELoss(),
        "cls_loss": nn.CrossEntropyLoss(),
        "student_scheduler": student_scheduler,
        "block_predictor_loss": nn.CrossEntropyLoss()
    }

def init_weights(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)
    elif isinstance(model, nn.LayerNorm):
        torch.nn.init.ones_(model.weight)
        torch.nn.init.zeros_(model.bias)

def create_loss_weights(occ_map):
    sorted_categ = sorted([int(key) for key in occ_map.keys()])
    sorted_occ = [occ_map[str(i)] for i in sorted_categ]
    sorted_tens = torch.tensor(sorted_occ)
    class_weights = 1.0 / (sorted_tens.float() + 1e-6)
    class_weights = class_weights / class_weights.sum() * 90
    extended = torch.cat([class_weights, class_weights[:1]])
    return extended

def create_finetune_model(geometry: str, jepa_embed_dim: int, class_parent: list[int], proj_dim: int, r_max: int, tau: int, margin: int):
    match geometry.lower():
        case "euclidean":
            return EuclideanEmbeddingSpaceHead(
                embed_dim=jepa_embed_dim,
                class_parent=class_parent,
                proj_dim=proj_dim,
                r_max=r_max,
                tau=tau,
                margin=margin
            )
        case "hyperbolic":
            return HyperbolicEmbeddingSpaceHead(
                embed_dim=jepa_embed_dim,
                class_parent=class_parent,
                proj_dim=proj_dim,
                r_max=r_max,
                tau=tau,
                margin=margin
            )
        case "spheric":
            return SphericEmbeddingSpaceHead(
                embed_dim=jepa_embed_dim,
                class_parent=class_parent,
                proj_dim=proj_dim,
                r_max=r_max,
                tau=tau,
                margin=margin
            )
        case _:
            return None