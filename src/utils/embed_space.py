import torch
import torch.nn as nn
import torch.nn.functional as F

class RootEmbeddingSpaceHead(nn.Module):

    def __init__(self, embed_dim, n_super, n_classes, proj_dim = 16, r_max = 2.0, tau = 1):
        super().__init__()

        # maximum length of a vector
        self.r_max = r_max

        self.tau = tau

        self.embed_proj = nn.Linear(embed_dim, proj_dim)
        self.class_nodes = nn.Parameter(0.1 * torch.randn(n_classes, proj_dim), requires_grad=True)
        self.category_nodes = nn.Parameter(0.1 * torch.randn(n_super, proj_dim), requires_grad=True)

        self.n_classes = n_classes

    def to_manifold(self, tens: torch.Tensor):
        """Transforms the input tensor into a specific embedding space (euclidean, hyperbolic or spheric)"""
        raise NotImplementedError

    def dist(self, x: torch.Tensor, y: torch.Tensor):
        """Calculates the distance between to input tensors in the corresponding embedding space"""
        raise NotImplementedError

    def clip(self, tens: torch.Tensor):
        norm = tens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return tens * (self.r_max / norm).clamp(max=1.0)

    def points(self, block_cls: torch.Tensor):
        # input is [B, N, D] , a batch of region CLS tokens
        cls_proj = self.to_manifold(self.clip(self.embed_proj(block_cls)))
        class_nodes_proj = self.to_manifold(self.clip(self.class_nodes))
        category_nodes_proj = self.to_manifold(self.clip(self.category_nodes))

        return cls_proj, class_nodes_proj, category_nodes_proj

    def class_logits(self, cls: torch.Tensor, class_nodes: torch.Tensor):
        return -self.dist(cls.unsqueeze(-2), class_nodes) ** 2 / self.tau

    def forward(self, block_cls: torch.Tensor):
        cls_proj, class_nodes, _ = self.points(block_cls)
        return self.class_logits(cls_proj, class_nodes)

    def losses(self, block_cls: torch.Tensor, labels: torch.Tensor):
        cls_proj, class_nodes, category_nodes = self.points(block_cls)
        logits = self.class_logits(cls_proj, class_nodes)

        loss_cls = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=self.n_classes)

        parent_logits = -self.dist(class_nodes.unsqueeze(1), category_nodes.unsqueeze(0)) ** 2 / self.tau

        
