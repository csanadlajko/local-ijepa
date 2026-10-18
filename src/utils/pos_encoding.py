## positional encoding credit to: https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer

import torch


def get_x_positions(num_patches, start_idx=0):
    num_patches_ = int(num_patches ** 0.5) # number of patches in the x dimension

    x_pos = torch.arange(start_idx, num_patches_ + start_idx)
    x_pos = x_pos.unsqueeze(0)
    x_pos = torch.repeat_interleave(x_pos, num_patches_, dim=0)
    x_pos = x_pos.reshape(-1)

    return x_pos

def get_y_positions(num_patches, start_idx=0):
    num_patches_ = int(num_patches ** 0.5)

    y_pos = torch.arange(start_idx, num_patches_+start_idx)
    y_pos = torch.repeat_interleave(y_pos, num_patches_, dim=0)

    return y_pos

def generate_sinusoidal_1d(position_sequence, embed_dim):
    embed_dim = embed_dim//2
    denominator = torch.pow(10000, torch.arange(0, embed_dim, 2) / embed_dim)

    pos_embedding = torch.zeros(1, position_sequence.shape[0], embed_dim)
    denominator = position_sequence / denominator
    pos_embedding[:, :, ::2] = torch.sin(denominator) ## even dimensions with sin
    pos_embedding[:, :, 1::2] = torch.cos(denominator) ## odd dimensions with cos

    return pos_embedding

def sinusoidal_pos_embedding2d(num_patches, embed_dim, device):
    x_positions = get_x_positions(num_patches).reshape(-1, 1)
    x_pos_embeddings = generate_sinusoidal_1d(x_positions, embed_dim)

    y_positions = get_y_positions(num_patches).reshape(-1, 1)
    y_pos_embeddings = generate_sinusoidal_1d(y_positions, embed_dim)

    pos_embed = torch.cat((x_pos_embeddings, y_pos_embeddings), -1).to(device)

    return pos_embed

def get_block_size(block_indices: list, total_patches: int) -> tuple[int, int]:
    row_len: int = total_patches ** 0.5
    start_idx: int = block_indices[0]
    targ_next_idx: int = start_idx + row_len
    i: int = 0
    while block_indices[i] != targ_next_idx:
        i+=1
    
    h: int = int(len(block_indices) / i)
    return h, i
    
def sinusoidal_block_pos_embedding2d(H, W, embed_dim, device):
    x_positions = torch.arange(W).repeat(H)
    y_positions = torch.arange(H).repeat_interleave(W)

    x_positions = x_positions.reshape(-1, 1)
    y_positions = y_positions.reshape(-1, 1)

    x_pos_embeddings = generate_sinusoidal_1d(x_positions, embed_dim)
    y_pos_embeddings = generate_sinusoidal_1d(y_positions, embed_dim)

    return torch.cat((x_pos_embeddings, y_pos_embeddings), -1).to(device)