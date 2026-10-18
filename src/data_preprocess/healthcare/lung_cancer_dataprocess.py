import json
import os
from abc import abstractmethod

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.parser.parser import parse_jepa_args

args = parse_jepa_args()

class LungCancerDataset:

    """
    Lung cancer dataset object obtained from Kaggle by Biplob Dey.\n
    [Source] https://www.kaggle.com/datasets/biplobdey/lung-and-colon-cancer
    """

    def __init__(self, input_dir: str, transform=None):
        self.input_dir = input_dir
        self.sub_dirs = os.listdir(input_dir) ## benign, adenocarcinoma and agressive cell
        self.transform = transform

        self.lc_class_map = {
            "aca": 0, ## Lung Adenocarcinoma - cancerous cells of the lung
            "bnt": 1, ## Lung Benign Tissue - healthy lung tissues
            "scc": 2 ## Lung Squamous Cell Carcinoma - aggressive lung cancer type
        }
        
        self.images = [
            f ## file name
            for sub_dir in self.sub_dirs ## in sub dirs
            if "lung" in sub_dir ## which are containing lung cancer images
            for f in os.listdir(os.path.join(input_dir, sub_dir)) ## for file in that folder
        ]
        self.labels = [self.lc_class_map[f.split("_")[1]] for f in self.images]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label: int = int(self.labels[idx])
        sub_folder_name: str = "_".join(self.images[idx].split("_")[:2])
        image_full_path: str = os.path.join(self.input_dir, sub_folder_name, self.images[idx])
        image = Image.open(image_full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class PDL1Dataset(Dataset):

    def __init__(self, img_dir: str, annotation_file: str, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(annotation_file) as f:
            coco_data = json.load(f)

        self.images = {f["id"]: f["file_name"] for f in coco_data["images"]}
        self.annotations = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        file_name = self.images[img_id]

        full_path = os.path.join(self.img_dir, file_name)
        img = Image.open(full_path).convert("RGB")

        w, h = img.size
        scale_x = args.image_size / w
        scale_y = args.image_size / h

        annotations = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x*scale_x, y*scale_y, w*scale_x, h*scale_y])
            labels.append(int(ann["category_id"]))

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
    
    @abstractmethod
    def collate_fn(batch):
        images = torch.stack([image[0] for image in batch])
        annotations = [ann[1] for ann in batch]
        return images, annotations