import json
from typing import Any

import torch

## ntc: non tumor cell
## ni (no information): no class in the current patch
## npdl1: pdl1 negative tumor cell
## ppdl1: pdl1 positive tumor cell

BLOCK_DATA_MAP: dict[str, int] = {
    "0": 0, # only no class in the block
    
    "1": 1, # only non tumor in the block
    "2": 2, # only pdl1 negative in the block
    "3": 3, # only positive pdl1 in the block

    "1_2": 4, # only non tumor and neg pdl1 in the block
    "1_3": 5, # only non tumor and pos pdl1 in the block
    "2_3": 6, # only pdl1 neg and pdl1 pos in the block
    "1_2_3": 7, # only non tumor, neg pdl1 and pos pdl1 in the block

    "0_1": 8, # only no class and non tumor in the block
    "0_2": 9, # only no class and pdl1 neg in the block
    "0_3": 10, # only no class and pdl1 pos in the block

    "0_1_2": 11, # only no class, no tumor and pdl1 neg
    "0_1_3": 12, # only no class, no tumor, pdl1 pos
    "0_2_3": 13, # only no class, pdl1 neg and pdl1 pos

    "0_1_2_3": 14 # every possible combination in the block
}

BLOCK_LABEL_MAP: dict[str, str] = {
    "0": "There are no identified cells in this block",
    
    "1": "This block only contains non-tumor cells", # only non tumor in the block
    "2": "This block only contains negative PDL1 cells", # only pdl1 negative in the block
    "3": "This block only contains positive PDL1 cells", # only positive pdl1 in the block

    "1_2": "This block only contains non-tumor and negative PDL1 cells", # only non tumor and neg pdl1 in the block
    "1_3": "This block only contains non-tumor and positive PDL1 cells", # only non tumor and pos pdl1 in the block
    "2_3": "This block only contains negative PDL1 and positive PDL1 cells", # only pdl1 neg and pdl1 pos in the block
    "1_2_3": "This block only contains non-tumor, negative PDL1 and positive PDL1 cells", # only non tumor, neg pdl1 and pos pdl1 in the block

    "0_1": "This block only contains unidentified and non-tumor cells", # only no class and non tumor in the block
    "0_2": "This block only contains unidentified and negative PDL1 cells", # only no class and pdl1 neg in the block
    "0_3": "This block only contains unidentified and positive PDL1 cells", # only no class and pdl1 pos in the block

    "0_1_2": "This block only contains unidentified, non-tumor and PDL1 negative cells", # only no class, no tumor and pdl1 neg
    "0_1_3": "This block only contains unidentified, non-tumor and PDL1 positivve cells", # only no class, no tumor, pdl1 pos
    "0_2_3": "This block only contains unidentified, PDL1 negative and PDL1 positive cells", # only no class, pdl1 neg and pdl1 pos

    "0_1_2_3": "This block contains unidentified, non-tumor, PDL1 negative and PDL1 positive cells" # every possible combination in the block
}

PATCH_DECODE_MAP: dict[str, str] = {
    "4": "1_3",
    "5": "2_3",
    "6": "1_2_3",
    "7": "1_2"
}

class PatchProcesser:

    # lowest abstarction level - patch level
    PATCH_DATA_MAP: dict[str, int] = {
        "ni": 0, # no class
        "3": 1, # non tumor
        "1": 2, # negative pd1
        "2": 3, # positive pdl1
        "1_3": 4, # negative pdl1 and non tumor cells
        "2_3": 5, # positive pdl1 and non tumor cells
        "1_2_3": 6, # all cells from all 3 classes
        "1_2": 7 # negative and positive pdl1 cells
    }

    def __init__(self, patch_size, logger):
        self.patch_size = patch_size
        self.logger = logger

    def _get_intersections(self, image_bboxes: list[list[int]], label_list: list[int], top_left_patch: tuple[int, int], bottom_right_patch: tuple[int, int]) -> int:
        assert len(image_bboxes) == len(label_list)
        jepa_classes = []
        px1, py1, px2, py2 = top_left_patch[0], top_left_patch[1], bottom_right_patch[0], bottom_right_patch[1]
        for i, bbox in enumerate(image_bboxes):
            bx1, by1, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
            bx2 = bw + bx1
            by2 = bh + by1
            # left-bottom-right-top coordinate inspection for intersection
            if (bx2 > px1) and (by1 < py2) and (bx1 < px2) and (by2 > py1):
                # if all the requirements are met above, the bbbox falls inside the current patch
                if label_list[i] not in jepa_classes:
                    jepa_classes.append(label_list[i])

        jepa_classes = sorted(jepa_classes)
        jepa_classes = [str(item) for item in jepa_classes]

        # return ground truth class integer (only one)
        if len(jepa_classes) != 0:
            jepa_classlist_string = "_".join(jepa_classes)
            return self.PATCH_DATA_MAP[jepa_classlist_string]
        return self.PATCH_DATA_MAP["ni"]

    def __call__(self, x: torch.Tensor, batch_bbox: list[dict[str, Any]]) -> torch.Tensor:
        B, C, H, W = x.shape

        batch_patch_metadata = []

        # for every image in a batch
        for b in range(B):
            # for every patch in a row
            bboxes: list = batch_bbox[b]["boxes"]
            labels: list = batch_bbox[b]["labels"]
            image_patch_metadata = []
            for y_cor in range(0, H, self.patch_size):
                # for ever patch in a column
                for x_cor in range(0, W, self.patch_size):
                    top_left: tuple[int, int] = (x_cor, y_cor)
                    bottom_right: tuple[int, int] = (x_cor+self.patch_size, y_cor+self.patch_size)
                    class_sum: int = self._get_intersections(
                        bboxes,
                        labels,
                        top_left,
                        bottom_right
                    )
                    image_patch_metadata.append(class_sum)
            batch_patch_metadata.append(image_patch_metadata)

        # a tensor containing the class information for every patch in every image in the batch
        # shape [B, N] where N is the corresponding class to every patch
        batch_md_tensor = torch.tensor(batch_patch_metadata, device=x.device, dtype=torch.long)
        assert batch_md_tensor.shape[0] == B
        return batch_md_tensor

def get_block_class(all_patch_data: torch.Tensor, block_indices: torch.Tensor, batch_idx: int):

    # patch classes containing cell combinations
    complex_patch_classes = [4, 5, 6, 7]
    
    # tensor with shape [N], meaning there is class paired to every target patch
    patch_indices = all_patch_data[batch_idx].index_select(dim=0, index=block_indices)

    patch_idx_list = patch_indices.tolist()

    patch_idx_unfiltered = []

    for index in patch_idx_list:
        if index in complex_patch_classes:
            basic_classes = [int(item) for item in PATCH_DECODE_MAP[str(index)].split("_")]
            patch_idx_unfiltered.extend(basic_classes)
        else:
            patch_idx_unfiltered.append(index)

    patch_idx_unfiltered = torch.tensor(patch_idx_unfiltered, device=all_patch_data.device)

    # get only unique classes
    unique_classes: torch.Tensor = torch.unique(patch_idx_unfiltered)

    # convert to list
    unique_sorted = sorted(unique_classes.tolist())
    unique_sorted = [str(item) for item in unique_sorted]

    # join so the indexing can be done
    classlist_string = "_".join(unique_sorted)

    # return corresponding summary class integer
    return torch.tensor([BLOCK_DATA_MAP[classlist_string]], device=all_patch_data.device)

class BlockProcessor:

    def __init__(self, patch_size, image_size, logger):
        self.patch_size = patch_size
        self.image_size = image_size
        self.categ_map = self._convert_coco_classes()
        self.logger = logger

    def _get_block_corner_coordinates(self, target_block: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        num_patches_per_side = self.image_size // self.patch_size

        top_left_patch_idx = torch.min(target_block).item()
        bottom_right_patch_idx = torch.max(target_block).item()

        y_pixel_step_1 = top_left_patch_idx // num_patches_per_side
        x_pixel_step_1 = top_left_patch_idx % num_patches_per_side

        top_left_pixel_coordinates = (x_pixel_step_1 * self.patch_size, y_pixel_step_1 * self.patch_size)
        
        y_pixel_step_2 = bottom_right_patch_idx // num_patches_per_side
        x_pixel_step_2 = bottom_right_patch_idx % num_patches_per_side

        _top_left_last_idx = (x_pixel_step_2 * self.patch_size, y_pixel_step_2 * self.patch_size)
        bottom_right_pixel_coordinates = (_top_left_last_idx[0] + self.patch_size, _top_left_last_idx[1] + self.patch_size)

        return top_left_pixel_coordinates, bottom_right_pixel_coordinates
    
    def _get_intersection_pct(self, target_block: torch.Tensor, box_coordinates: list[int]):
        top_left_block, bottom_right_block = self._get_block_corner_coordinates(target_block)

        x_t_1, y_t_1 = top_left_block[0], top_left_block[1]
        x_t_2, y_t_2 = bottom_right_block[0], bottom_right_block[1]

        x_b_1, y_b_1 = box_coordinates[0], box_coordinates[1]
        x_b_2 = x_b_1 + box_coordinates[2]
        y_b_2 = y_b_1 + box_coordinates[3]

        x_left = max(x_t_1, x_b_1)
        x_right = min(x_t_2, x_b_2)
        y_top = max(y_t_1, y_b_1)
        y_bottom = min(y_t_2, y_b_2)

        # check intersection
        if x_right > x_left and y_top < y_bottom:
            # check intersection percentage
            intersection_area = (x_right-x_left) * (y_bottom-y_top)
            bbox_area = box_coordinates[2] * box_coordinates[3]
            return intersection_area / bbox_area
        else:
            return 0

    def _get_largest_bbox_intersection(self, target_block: torch.Tensor, bbox_list: list[list[int]], int_labels: list[int], str_labels: list[str]):
        assert len(int_labels) == len(bbox_list)
        index = None
        max_pct = 0
        
        for i, bbox in enumerate(bbox_list):
            pct = self._get_intersection_pct(target_block, bbox)
            if pct > max_pct:
                max_pct = pct
                index = i

        if index is not None:
            return self.categ_map[str(int_labels[index])], str_labels[index]
        
        return 80, "background"
    
    def _convert_coco_classes(self):
        with open("src/data_preprocess/coco/categories.json") as f:
            categories = json.load(f)

        categ_map = { str(int(categ["id"]) - 1): idx for idx, categ in enumerate(categories["categories"]) }

        return categ_map

    def __call__(self, batch_bbox_list: list[list[list[int]]], batch_target_block: list[list[torch.Tensor]], string_labels: list[list[str]], int_labels: list[list[int]]):
        assert len(batch_bbox_list) == len(batch_target_block)
        assert len(string_labels) == len(int_labels)
        assert len(int_labels) == len(batch_bbox_list)

        total_data_int = []
        total_data_string = []

        for i, image_bbox_list in enumerate(batch_bbox_list):
            batch_classes_int = []
            batch_classes_string = []
            for target_indices in batch_target_block[i]:
                gt_block_class, gt_block_name = self._get_largest_bbox_intersection(target_indices, image_bbox_list, int_labels[i], string_labels[i])
                batch_classes_string.append(gt_block_name)
                batch_classes_int.append(gt_block_class)
            total_data_int.append(batch_classes_int)
            total_data_string.append(batch_classes_string)

        tens_int_classes = torch.tensor(total_data_int, device=batch_target_block[0][0].device)

        assert tens_int_classes.shape[0] == len(batch_bbox_list)
        assert tens_int_classes.shape[1] == len(batch_target_block[0])

        return tens_int_classes, total_data_string

def compute_weights(targets_flat, num_classes=81):
    counts = torch.bincount(targets_flat, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights