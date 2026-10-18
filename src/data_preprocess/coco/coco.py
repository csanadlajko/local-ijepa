import json
import os
from abc import abstractmethod

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.parser.parser import parse_jepa_args

args = parse_jepa_args()

class COCODataset(Dataset):

    def __init__(self, img_dir: str, annotation_json: str, transforms = None):
        self.img_dir = img_dir
        self.annotation_json = annotation_json
        self.transforms = transforms

        with open(annotation_json) as f:
            coco_data = json.load(f)
    
        self.category_map = { str(categ["id"]): categ["name"] for categ in coco_data["categories"] }
        self.images = {f["id"]: f["file_name"] for f in coco_data["images"]}
        self.annotations = {}
        self.class_occ_map = {}
        super_categ_map = self._create_superctageory_map(coco_data["categories"])

        # len(80) list containing supercategory ids for every class
        self.class_parent = [super_categ_map[item["supercategory"]] for item in coco_data["categories"]]
        for ann in coco_data["annotations"]:
            # connect annotations to images
            img_id = ann["image_id"]
            subtr_cat_id = int(ann["category_id"]) - 1
            if str(subtr_cat_id) in self.class_occ_map.keys():
                self.class_occ_map[str(subtr_cat_id)] += 1
            else:
                self.class_occ_map[str(subtr_cat_id)] = 0
            self.annotations.setdefault(img_id, []).append(ann)

        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        file_name = self.images[img_id]

        full_path = os.path.join(self.img_dir, file_name)
        image = Image.open(full_path).convert("RGB")

        w, h = image.size

        # scale to resized image
        scale_x = args.image_size / w
        scale_y = args.image_size / h

        annotations = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        string_labels = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x*scale_x, y*scale_y, w*scale_x, h*scale_y])
            subtracted_category: int = ann["category_id"] - 1
            labels.append(subtracted_category)
            string_labels.append(self.category_map[str(ann["category_id"])])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "string_labels": string_labels
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
    
    @abstractmethod
    def collate_fn(batch):
        images = torch.stack([image[0] for image in batch])
        annotations = [ann[1] for ann in batch]
        return images, annotations

    def _create_superctageory_map(self, categories) -> dict[str, int]:
        categ_id: int = 0
        categ_map: dict[str, int] = {}
        for item in categories:
            if item["supercategory"] not in categ_map:
                categ_map[item["supercategory"]] = categ_id
                categ_id+=1
        return categ_map