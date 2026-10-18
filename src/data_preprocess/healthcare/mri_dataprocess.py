import os

from PIL import Image
from torch.utils.data import Dataset


class MRIImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.imgs = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]

        self.labels = [f.split('_')[-1].split('.')[0] for f in self.imgs]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        image = Image.open(img_path).convert('RGB')
        
        label = int(self.labels[idx])-1 ## -1 for CE loss

        if self.transform is not None:
            image = self.transform(image)

        return image, label