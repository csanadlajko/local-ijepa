import torch
from torch import nn

from ..utils.masking import apply_mask
from ..utils.pos_encoding import (
    get_block_size,
    sinusoidal_block_pos_embedding2d,
    sinusoidal_pos_embedding2d,
)
from .vit import TransformerEncoder


class ViTPredictor(nn.Module):

    def __init__(
            self, 
            num_patches,
            device,
            num_targets,
            logger,
            embed_dim=256,
            pred_dim=None, 
            depth=6,
            num_heads=8, 
            drop_rate=0.1,
            num_classes=10,
            tokenizer=None, 
            text_encoder=None
        ):
        super().__init__()
        if pred_dim is None:
            pred_dim = embed_dim
        self.num_targets = num_targets
        self.num_patches = num_patches

        self.logger = logger

        self.predictor_embed = nn.Linear(embed_dim, pred_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim), requires_grad=True) # learnable parameters to predict masked region
        # CLS token for target blocks
        self.target_block_cls = nn.Parameter(torch.zeros(1, 1, pred_dim), requires_grad=True)

        self.pred_pos_embed = sinusoidal_pos_embedding2d(num_patches, embed_dim, device)

        self.pred_blocks = nn.Sequential(*[
            TransformerEncoder(
                num_heads=num_heads,
                embed_dim=pred_dim,
                mlp_dim=256,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        self.target_block_transformer = nn.Sequential(*[
            TransformerEncoder(
                num_heads=num_heads,
                embed_dim=pred_dim,
                mlp_dim=256,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        self.label_to_embed = nn.Sequential(
            nn.Linear(384, pred_dim), ## depends on the embedding size of the text encoder
            nn.LayerNorm(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, embed_dim)
        )

        self.cls_fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.cls_fc2 = nn.Linear(embed_dim // 2, num_classes)
        self.cls_position = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        nn.init.xavier_uniform_(self.cls_fc1.weight)
        nn.init.zeros_(self.cls_fc1.bias)
        nn.init.xavier_uniform_(self.cls_fc2.weight)
        nn.init.zeros_(self.cls_fc2.bias)

        self.device = device

        self.cls_head = nn.Sequential(
            self.cls_fc1,
            nn.GELU(),
            nn.Dropout(drop_rate),
            self.cls_fc2
        )

        self.post_pred_mhsa = nn.MultiheadAttention(pred_dim, num_heads, drop_rate, batch_first=True)

        self.predictor_norm = nn.LayerNorm(pred_dim)
        self.block_cls_norm = nn.LayerNorm(embed_dim)
        self.predictor_proj = nn.Linear(pred_dim, embed_dim) # back to encoder dimension

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def forward(self, x, context_mask, target_mask, labels: list[list[str]], multimodal: bool, local_cls: bool = False, return_cls_only = False, cell_mask=False, block_cls=False):
        ## x includes cls token on index 0
 
        B = x.size(0)
        N_ctx = x.size(1)
        
        x = self.predictor_embed(x)

        h, w = get_block_size(list(target_mask[0][0]), self.num_patches)
        block_cls_pos_enc = sinusoidal_block_pos_embedding2d(h, w, x.shape[2], x.device).repeat(B, 1, 1)
        cls_pos_enc = self.cls_position.repeat(B, 1, 1)
        block_final_pos = torch.cat([block_cls_pos_enc, cls_pos_enc], dim=1)
        
        target_positions, pred_target_attn = apply_mask(self.pred_pos_embed.repeat(B, 1, 1), target_mask, predictor=True, use_padding=cell_mask)
        num_target_tokens = target_positions.size(1)
        # in case of padding, the learnable mask tokens are containing the paddings...
        mask_tokens = self.mask_token.repeat(target_positions.size(0), num_target_tokens, 1)
        block_cls_embeddings = self.target_block_cls.repeat(B, self.num_targets, 1)
        mask_tokens = mask_tokens + target_positions
        
        x = torch.cat([x, mask_tokens], dim=1) # full predicted image with cls token on index 0

        # run attention on whole image
        for block in self.pred_blocks:
            x = block(x, None)

        x = self.predictor_norm(x)

        predicted_tokens = x[:, N_ctx:]

        block_cls_tokens_raw = []
        for block in range(self.num_targets):
            block_end = int((block + 1) * (predicted_tokens.shape[1] / self.num_targets))
            block_start = int(((block + 1) * (predicted_tokens.shape[1] / self.num_targets)) - (predicted_tokens.shape[1] / self.num_targets))

            batch_block = predicted_tokens[:, block_start:block_end, :]
            corresp_cls = block_cls_embeddings[:, block, :]
            corresp_cls = corresp_cls.unsqueeze(1)

            cls_extended = torch.cat([corresp_cls, batch_block], dim=1)
            cls_extended = cls_extended + block_final_pos

            for att_block in self.target_block_transformer:
                cls_extended = att_block(cls_extended, None)

            if local_cls:  # misleading parameter -> stands for multimodal run....
                current_classes = [image_label[block] for image_label in labels] # len(batch)
                labels_for_block = [f"this block represents a {c}" for c in current_classes]
                label_tokens: dict[str, torch.Tensor] = self.tokenizer(labels_for_block, return_tensors='pt', padding=True)
                label_tokens = {k: v.to(self.device) for k, v in label_tokens.items()}
                enc_labels = self.text_encoder(**label_tokens, output_hidden_states=True)
                enc_labels = enc_labels.hidden_states[-1]

                enc_labels = enc_labels.to(self.device)

                enc_labels = self.label_to_embed(enc_labels)

                pred_attended, _ = self.post_pred_mhsa(cls_extended, enc_labels, enc_labels)

                cls_extended = cls_extended + pred_attended
            
            # append only the block cls token
            cls_only = cls_extended[:, 0, :].unsqueeze(1)
            block_cls_tokens_raw.append(cls_only)
        
        # should be a tensor with a shape [B, self.num_targets, embed_dim]
        block_cls_tokens_cat = torch.cat(block_cls_tokens_raw, dim=1)
        block_cls_tokens_cat = self.block_cls_norm(block_cls_tokens_cat)

        predicted_tokens = self.predictor_proj(predicted_tokens)
        
        return predicted_tokens, block_cls_tokens_cat

class CellTypePredictor(nn.Module):

    def __init__(self, num_classes=8, embed_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.pred_layer = nn.Linear(embed_dim, num_classes)

        # self.pred_layer = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, num_classes)
        # )

    def forward(self, x: torch.Tensor, target_patch_indices: list[list[torch.Tensor]], gt_patch_classes: torch.Tensor, loss_fn):
        """
        :x predicted target patch embeddings with shape [B, N, D] where N is the (same) padded number of target patches in the batch
        :target_patch_indices list of length B containing lists of one index tensor each
        :gt_patch_classes ground-truth class labels for the target patches, shape [B, N] where N is the maximum number of patches in an image

        depracated ... was used on the PD-L1 dataset
        """
        
        total_losses = []
        total_samples = 0
        total_correct = 0

        # for every image in the batch
        for b_idx, batch in enumerate(target_patch_indices):
            # for every target patch index in an image
            patch_indices = torch.cat(batch)
            
            gt_class = gt_patch_classes[b_idx].index_select(dim=0, index=patch_indices)
            # only select the non padded indices from tensor x
            pred_classes = self.pred_layer(x[b_idx, :, :])
            loss = loss_fn(pred_classes, gt_class)
            pred_labels = torch.argmax(pred_classes, dim=1)

            correct = (pred_labels == gt_class).sum().item()

            total_correct += correct
            total_samples += gt_class.numel()
            total_losses.append(loss)

        avg_loss = sum(total_losses) / len(total_losses)
        avg_accuracy = (total_correct / total_samples) * 100

        return avg_loss, avg_accuracy


class BlockTypePredictor(nn.Module):

    def __init__(self, logger, num_classes=15, embed_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.logger = logger
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, predicted_block_cls: torch.Tensor):
        # from [B, N_cls, D] to [B, N_cls, T]
        pred_classes = self.prediction_head(predicted_block_cls)
        return pred_classes