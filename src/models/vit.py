import torch
from torch import nn

from ..utils.masking import apply_mask
from ..utils.pos_encoding import sinusoidal_pos_embedding2d


class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PatchEmbed(nn.Module):
    
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        
    def forward(self, x: torch.Tensor, cls=True):
        B, C, H, W = x.shape # -> should be B - N (total_num_of_patches) - D (embed dim from conv2d) -> (16, 3, 128, 128)
        x = self.proj(x).flatten(2).transpose(1, 2) ## (16, 256, 8, 8) -> (16, 256, 64) -> (16, 64, 256)
        x = x + sinusoidal_pos_embedding2d(self.num_patches, self.embed_dim, x.device)
        if not cls:
            ## return if no cls token is needed
            return x
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, dim=0), x), dim=1) # concat on dim 1: B,N,D -> B,N+1,D (cls token)
        return x
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, num_heads, embed_dim, mlp_dim, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_dim, out_features=embed_dim, act_layer=nn.GELU, drop=drop)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, pad):
        attn_output, _ = self.att(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x)
        )
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
    
class VisionTransformer(nn.Module):
    
    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_heads, depth, mlp_dim, drop_rate, logger, num_classes=None):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.encoder = nn.Sequential(*[
            TransformerEncoder(
                num_heads=num_heads,
                embed_dim=embed_dim,
                mlp_dim=mlp_dim,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        self.logger = logger
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, masks=None, return_cls_only=False, cell_mask=False, cls=True):
        x = self.patch_embed(x, cls) # patch embed and pos encoding
        attn_mask = None
        if masks is not None and not return_cls_only and not cell_mask:
            x, attn_mask = apply_mask(x, masks, predictor=True) # only needed when entering with student model
        elif masks is not None and cell_mask:
            ## used when pdl1 cell context mask is given
            x, attn_mask = apply_mask(x, masks, predictor=True, use_padding=True)

        for block in self.encoder:
            x = block(x, None)

        x = self.norm(x)
        
        return x, attn_mask