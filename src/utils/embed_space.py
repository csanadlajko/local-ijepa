import torch
import torch.nn as nn


class RootEmbeddingSpace(nn.Module):

    def __init__(self, input_dim, num_classes, num_super_categories, projection_dim = 16):
        super().__init__()

        self.embed_proj = nn.Linear(input_dim, projection_dim)

        self.class_nodes = nn.Parameter(torch.zeros(num_classes, projection_dim), requires_grad=True)
        self.category_nodes = nn.Parameter(torch.zeros(num_super_categories, projection_dim), requires_grad=True)

    def cls_to_class_node_distance(self):
        pass

    def class_node_to_category_node_distance(self):
        pass

    def depth(self):
        pass


class EuclideanEmbeddingSpace(RootEmbeddingSpace):

    def __init__(self, input_dim, num_classes, num_super_categories, projection_dim=16):
        super().__init__(input_dim, num_classes, num_super_categories, projection_dim)

    def cls_to_class_node_distance(self):
        pass
    
    def class_node_to_category_node_distance(self):
        pass

    def depth(self):
        pass


class HyperbolicEmbeddingSpace(RootEmbeddingSpace):

    def __init__(self, input_dim, num_classes, num_super_categories, projection_dim=16):
        super().__init__(input_dim, num_classes, num_super_categories, projection_dim)

    def cls_to_class_node_distance(self):
        pass
    
    def class_node_to_category_node_distance(self):
        pass

    def depth(self):
        pass


class SphericEmbeddingSpace(RootEmbeddingSpace):

    def __init__(self, input_dim, num_classes, num_super_categories, projection_dim=16):
        super().__init__(input_dim, num_classes, num_super_categories, projection_dim)

    def cls_to_class_node_distance(self):
        pass

    def class_node_to_category_node_distance(self):
        pass

    def depth(self):
        pass