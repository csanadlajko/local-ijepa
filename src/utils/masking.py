import math
import random
from multiprocessing import Value
from typing import Any

import torch
import torch.nn.functional as F

from src.parser.parser import parse_jepa_args

args = parse_jepa_args()

DEBUG = True if args.debug == "y" else False

class CellMask:

    """
    Mask a cell by the following logic: \n
        - Check patch index of (x, y) by bounding box,
        - Check patch index of (x, y - h) by bounding box,
        - Check patch index of (x + w, y) by bounding box,
        - Check patch index of (x + w, y - h) by bounding box.\n
    Afterwards mask these patches as they will give the local representations.
    """

    def __init__(
            self,
            device,
            input_size=(args.image_size, args.image_size),
            patch_size=args.patch_size
        ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.patch_size = patch_size
        self.num_pathes = self.height * self.width
        self.device = device

    def _get_patch_indices_by_coordinates(
            self, 
            annotation_batch: list[dict[str, Any]] ## list of annotations per batch
        ) -> list[list]:
        
        """
        Returns the flattened patch indices where the bounding box overlaps.
        """
        ## a list for containing all the patch indices for every image in the batch
        batch_patch_indices: list[list] = []
        batch_labels: list[list] = []
        for annotation in annotation_batch:
            patch_indices = set()
            label_map = {}
            for i, box in enumerate(annotation["boxes"]):
                x, y, w, h = box ## x, y top left; w, h width-height (not bottom right corner!!!)

                ## transform from regular coordinates to patch coordinates
                top_left_patch_index = (x // self.patch_size, y // self.patch_size)
                bottom_right_patch_index = ((x+w) // self.patch_size, (y+h) // self.patch_size)

                x_0, y_0 = top_left_patch_index
                x_1, y_1 = bottom_right_patch_index

                ## cover the whole block by top left and bottom right patch indices
                for y in range(int(y_0), int(y_1)):
                    for x in range(int(x_0), int(x_1)):
                        ## transform from patch coordinates to flattened patch index
                        idx = (self.width * y) + x
                        patch_indices.add(idx)
                        label_map[idx] = annotation["labels"][i]
            batch_patch_indices.append(list(patch_indices))
            batch_labels.append(list(label_map.values()))
        return batch_patch_indices, batch_labels
    
    def __call__(self, cell_percentage, bbox):
        """
        Covers ``cell_percentage`` of the given cells, uses the rest as the context space.
        """
        ## len(batch_size) list (B, K) where K is the number of found patches per image
        ## contains patch indices for each image in the batch
        all_target_cells, target_labels = self._get_patch_indices_by_coordinates(bbox)
        pct = float(cell_percentage / 100)
        ## (B, K) filtered list for target patches
        selected_target_patches: list = []
        selected_target_labels: list = []
        ## (B, K) filtered list for context pacthes
        remaining_context_patches: list = []
        remaining_context_labels: list = []
        ## might not be in sync
        for image, labels in zip(all_target_cells, target_labels):
            selected_number = int(len(image) * pct)
            sample_indices = random.sample(range(len(image)), selected_number)
            sample_target = [image[i] for i in sample_indices]
            target_labels = [labels[i] for i in sample_indices]
            selected_target_labels.append(target_labels)
            selected_target_patches.append([torch.tensor(sample_target)])
            sample_set = set(sample_target)
            label_set = set(target_labels)
            unchoosen = [idx for idx in image if idx not in sample_set]
            context_labels = [label for label in labels if label not in label_set]
            remaining_context_labels.append(context_labels)
            remaining_context_patches.append([torch.tensor(unchoosen)])

        return {
            "context_patch_indices": remaining_context_patches,
            "context_patch_labels": remaining_context_labels,
            "target_patch_indices": selected_target_patches,
            "target_patch_labels": selected_target_labels
        }


class Mask:
    
    def __init__(
        self,
        device,
        input_size=(args.image_size, args.image_size),
        patch_size=args.patch_size,
        nctx=1,
        ntarg=args.num_target,
        targ_mask_scale=(0.02, 0.05),
        ctx_mask_scale=(0.2, 0.9),
        aspect_ratio=(0.75, 1.5),
        min_keep=4,
        max_tries=20
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.patch_size = patch_size
        self.nctx = nctx
        self.ntarg = ntarg
        self.targ_mask_scale = targ_mask_scale
        self.ctx_mask_scale = ctx_mask_scale
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep
        self.max_tries = max_tries
        self.device = device
        self.iteration_counter = Value('i', -1)
        
    def step(self):
        """
        Increments ``i`` for every image, resulting in different context and target blocks.
        """
        i = self.iteration_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
        
    def _sample_block_size(self, generator, scale, ascpect_ratio_scale):
        u = torch.rand(1, generator=generator).item()
        
        # block size scale
        min_s, max_s = scale 
        
        # ratio of block size / total patch number
        frac = min_s + u * (max_s - min_s)
        
        # number of patches in the block
        num_patches = int(self.height*self.width * frac)
        
        # provide the minimum amount of patches per block
        num_patches = max(num_patches, self.min_keep)
        
        v = torch.rand(1, generator=generator).item()
        min_asp_ratio, max_asp_ratio = ascpect_ratio_scale
        aspect_ratio = min_asp_ratio + v * (max_asp_ratio - min_asp_ratio)
        
        h = int(round(math.sqrt(num_patches * aspect_ratio)))
        w = int(round(math.sqrt(num_patches / aspect_ratio)))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        h = max(h, 1)
        w = max(w, 1)
        return h, w
    
    def _place_block_without_overlap(self, h, w, occ):
        H, W = occ.shape
        tries = 0
        found_target = 0
        total_target = h*w
        while found_target <= total_target or tries > self.max_tries:
            top = torch.randint(0, H - h + 1, (1,)).item()
            left = torch.randint(0, W- w + 1, (1,)).item()
            cut_region = occ[top:top+h, left:left+w]

            if torch.count_nonzero(cut_region) == 0:
                mask = torch.zeros((H, W), dtype=torch.int32)
                mask[top:top+h, left:left+w] = 1
                occ[top:top+h, left:left+w] = 1
                idx = torch.nonzero(mask.flatten(), as_tuple=False).squeeze()
                found_target = idx.numel()

                if idx.numel() == total_target:
                    return idx, occ
            tries += 1
        
        idx = torch.randperm(H*W)[:max(self.min_keep, h*w // 2)]
        return idx, occ
    
    def __call__(self, batch, bbox_list=None, id_only=True):
        B = len(batch)
        if isinstance(batch, torch.Tensor):
            collated_batch = batch
        else:
            collated_batch = torch.utils.data.default_collate(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        ctx_h, ctx_w = self._sample_block_size(g, self.ctx_mask_scale, (1., 1.))
        target_h, target_w = self._sample_block_size(g, self.targ_mask_scale, self.aspect_ratio)
        
        all_mask_ctx, all_mask_target = [], []

        max_patch = self.width * self.height
        ctx_size = int(max_patch * 0.8)

        for b in range(B):
            occ = torch.zeros((self.height, self.width), dtype=torch.int32)
            
            target_mask = []

            if bbox_list is not None and b < len(bbox_list):
                bboxes = bbox_list[b]
                centroids = []
                for box in bboxes:
                    x, y, w, h = box
                    cx = int((x + w / 2) / self.patch_size)
                    cy = int((y + h / 2) / self.patch_size)
                    centroids.append((cx, cy))
                if len(centroids) > self.ntarg:
                    centroids = random.sample(centroids, self.ntarg)
                elif len(centroids) < self.ntarg:
                    for _ in range(self.ntarg - len(centroids)):
                        cx = random.randint(0, self.width - 1)
                        cy = random.randint(0, self.height - 1)
                        centroids.append((cx, cy))
            else:
                centroids = None

            needed = max_patch - ctx_size
            for i in range(self.ntarg):
                if centroids is not None and i < len(centroids):
                    cx, cy = centroids[i]
                    top = max(0, cy - target_h // 2)
                    left = max(0, cx - target_w // 2)
                    top = min(top, self.height - target_h)
                    left = min(left, self.width - target_w)
                else:
                    top = torch.randint(0, self.height - target_h + 1, (1,)).item()
                    left = torch.randint(0, self.width - target_w + 1, (1,)).item()
                
                cut_region = occ[top:top+target_h, left:left+target_w]
                if torch.count_nonzero(cut_region) == 0:
                    mask = torch.zeros((self.height, self.width), dtype=torch.int32)
                    mask[top:top+target_h, left:left+target_w] = 1
                    occ[top:top+target_h, left:left+target_w] = 1
                    idx = torch.nonzero(mask.flatten(), as_tuple=False).squeeze()
                    target_mask.append(idx.to(self.device))
                else:
                    idx, occ = self._place_block_without_overlap(target_h, target_w, occ)
                    target_mask.append(idx.to(self.device))
            
            free = (occ == 0).to(torch.int32)
            
            tries = 0
            cmask = None
            
            while tries < self.max_tries:
                top = torch.randint(0, self.height - ctx_h + 1, (1,)).item()
                left = torch.randint(0, self.width - ctx_w + 1, (1,)).item()
                region = free[top:top+ctx_h, left:left+ctx_w]
                if torch.all(region == 1):
                    cmask2d = torch.zeros((self.height, self.width), dtype=torch.int32)
                    cmask2d[top:top+ctx_h, left:left+ctx_w] = 1
                    cmask = torch.nonzero(cmask2d.flatten(), as_tuple=False).squeeze()
                    break
                tries += 1
            
            free_idx = torch.nonzero(free.flatten(), as_tuple=False).view(-1)

            target_size = ctx_h * ctx_w
            
            if cmask is None:
                if free_idx.numel() == 0:
                    cmask = torch.randperm(self.height * self.width)[:target_size]
                else:
                    perm = torch.randperm(free_idx.numel())
                    take = min(target_size, free_idx.numel())
                    cmask = free_idx[perm[:take]]

            while needed > 0 and free_idx.numel() > 0:
                if cmask is not None and cmask.numel() > 0:
                    free_idx = free_idx[~torch.isin(free_idx, cmask)]
                
                if free_idx.numel() == 0:
                    break
                
                perm = torch.randperm(free_idx.numel())
                take = min(needed, free_idx.numel())
                additional = free_idx[perm[:take]]
                
                if cmask is None:
                    cmask = additional
                else:
                    cmask = torch.cat([cmask, additional])
                
                needed -= take
              
            all_mask_target.append(target_mask)
            cmask = cmask.to(self.device)
            all_mask_ctx.append(cmask)
            
        if id_only:
            return all_mask_ctx, all_mask_target
        
        masked_ctx_batch = batch.clone()
        masked_target_batch = batch.clone()
        
        for i in range(B):
            ctx_idx = all_mask_ctx[i][0] + 1
            targ_id_list = all_mask_target[i]
            
            target_idx = torch.cat([idx for idx in targ_id_list]) + 1
            masked_ctx_batch[i, target_idx, :] = 0  # Mask target tokens
            
            masked_target_batch[i, ctx_idx, :] = 0  # Mask context tokens
            
        return collated_batch, masked_ctx_batch, masked_target_batch
    
def apply_mask(x: torch.Tensor, mask_indices: list[torch.Tensor], predictor=False, use_padding=False):
    if isinstance(mask_indices, list):
        all_masked_tokens = []
        for i, mask_idx in enumerate(mask_indices):
            if isinstance(mask_idx, list):
                # enter when selecting target blocks
                all_idx = torch.cat(mask_idx)
                conc=False # dont concat when using teacher, only use N patches for masking
            else:
                # enter when selecting context blocks
                all_idx = mask_idx
                if predictor is False:
                    conc=True # concat when selecting context blocks for vit and vit predictor
                else:
                    conc=False
            if all_idx.numel() > 0:
                if predictor is False:
                    patch_idx = (all_idx+1).to(x.device) ## shift indices to right because cls
                else:
                    patch_idx = all_idx.to(x.device) ## dont shift when using predictor -> not predicting cls token
                ## concat cls when working with context blocks otherwise only shift indices to right
                if conc:
                    indices = torch.cat([torch.tensor([0], device=x.device), patch_idx])
                else:
                    indices = patch_idx
                masked_tokens = x[i].index_select(dim=0, index=indices)
                all_masked_tokens.append(masked_tokens.unsqueeze(0))
        ## ??? empty batches are possible ????
        if use_padding is False:
            return torch.cat(all_masked_tokens, dim=0), None
        else:
            # get maximum padding
            N_max = max(t.shape[1] for t in all_masked_tokens)

            # pad all embeddings to the longest patch length (D_left, D_right, N_left, N_right)
            all_masked_tokens = [
                F.pad(input=t, pad=(0, 0, 0, N_max - t.shape[1]), value=-100)
                for t in all_masked_tokens
            ]

            batch_masked = torch.cat(all_masked_tokens, dim=0) 
            # create att. mask -> true if true index is given, false if false index is given
            # compulsory for MHSA
            attention_mask = (batch_masked[:, :, 0] == -100)

            return batch_masked, attention_mask
    else:
        return x.index_select(1, mask_indices), None
