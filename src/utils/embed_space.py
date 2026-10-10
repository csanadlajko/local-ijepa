import torch
import torch.nn as nn
import torch.nn.functional as F

class RootEmbeddingSpaceHead(nn.Module):

    def __init__(self, embed_dim, class_parent, proj_dim = 16, r_max = 2.0, tau = 1, margin = 0.5):
        super().__init__()

        # maximum length of a vector
        self.r_max = r_max
        self.margin = margin

        self.tau = tau

        # len(n_class) list where every item represents the current class's super category
        self.register_buffer("class_parent", torch.as_tensor(class_parent, dtype=torch.long))

        self.n_classes = len(class_parent)
        self.n_super = int(self.class_parent.max()) + 1

        self.embed_proj = nn.Linear(embed_dim, proj_dim)
        self.class_nodes = nn.Parameter(0.1 * torch.randn(self.n_classes, proj_dim), requires_grad=True)
        self.category_nodes = nn.Parameter(0.1 * torch.randn(self.n_super, proj_dim), requires_grad=True)

    def to_manifold(self, tens: torch.Tensor):
        """Transforms the input tensor into a specific embedding space (euclidean, hyperbolic or spheric)"""
        raise NotImplementedError

    def dist(self, x: torch.Tensor, y: torch.Tensor):
        """Calculates the distance between to input tensors in the corresponding embedding space"""
        raise NotImplementedError

    def radius(self, tens: torch.Tensor):
        """Distance of a point from the root of the manifold"""
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
        # point embeddings
        cls_proj, class_nodes, category_nodes = self.points(block_cls)

        # cls logits
        # from embeddings to logits calculated from distance between CLS token and class prototype
        # return [16, 5, 80] where 80 is the logit value from the distance calculation 
        logits = self.class_logits(cls_proj, class_nodes)

        # CE on [80, 80] and [80] ([16, 5] label tensor flattened, classes for every block in the batch) tensor
        loss_cls = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=self.n_classes)

        # distance between [80, 1, 16] and [1, 12, 16]
        # return [80, 12, 16] ([CL, CA, D]), for every class CL, the distance between category CA is D -> creates logits as well 
        parent_logits = -self.dist(class_nodes.unsqueeze(1), category_nodes.unsqueeze(0)) ** 2 / self.tau

        # GT (self.class_parent) is the list of the actualy supercategories of all the classes
        loss_parent = F.cross_entropy(parent_logits, self.class_parent)
        
        # parent embeddings should be closer to the root with at least self.margin amount, than the class nodes
        # if gap is negative, it means th parent + radius distance is smaller (closer) -> relu gives 0
        # if its larger than 0, it means the child is closer, relu returns its value as the loss (mean afterwards)
        gap = self.radius(category_nodes)[self.class_parent] + self.margin - self.radius(class_nodes)
        loss_depth = F.relu(gap).mean()

        return logits, {"cls": loss_cls, "parent": loss_parent, "depth": loss_depth}


class EuclideanEmbeddingSpaceHead(RootEmbeddingSpaceHead):

    def dist(self, x: torch.Tensor, y: torch.Tensor):
        return ((x - y) ** 2).sum(-1).add(1e-9).sqrt()
    
    def to_manifold(self, tens: torch.Tensor):
        return tens

    def radius(self, tens: torch.Tensor):
        return (tens ** 2).sum(-1).add(1e-9).sqrt()

class HyperbolicEmbdeddingSpaceHead(RootEmbeddingSpaceHead):

    def to_manifold(self, tens: torch.Tensor):
        """Transform a [B, N, D] tensor into lower dimension hyperbolic space"""
        norm = tens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        
        # expmap -> geodesic distance * direction
        return torch.tanh(norm) * tens / norm

    def dist(self, x: torch.Tensor, y: torch.Tensor):
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        den = ((1 - x_norm ** 2) * (1 - y_norm ** 2)).clamp_min(1e-6)
        return torch.arccosh((1 + 2 * ((x - y) ** 2).sum(-1) / den).clamp_min(1 + 1e-6))

    def radius(self, tens: torch.Tensor):
        norm = tens.norm(dim=-1)
        return 2 * torch.arctanh(norm)

class SphericEmbeddingSpaceHead(RootEmbeddingSpaceHead):

    def to_manifold(self, tens: torch.Tensor):
        return super().to_manifold(tens)

    def dist(self, x: torch.Tensor, y: torch.Tensor):
        return super().dist(x, y)

    def radius(self, tens: torch.Tensor):
        return super().radius(tens)