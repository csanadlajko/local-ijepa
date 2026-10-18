import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10dot1Dataset(Dataset):

    def __init__(self, data_path, label_path, transform=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]
        label = int(self.labels[index])

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, label