from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch
from .healthcare.mri_dataprocess import MRIImageDataset
from .healthcare.lung_cancer_dataprocess import LungCancerDataset, PDL1Dataset
from .coco.coco import COCODataset
from .cifar10dot1 import CIFAR10dot1Dataset
from ..parser.parser import parse_jepa_args

args = parse_jepa_args()

train_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomInvert(0.3),
    # transforms.RandomHorizontalFlip(p=0.6),
    # transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                     std=[0.247, 0.243, 0.261])
])


test_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

def get_pdl1_dataset(input_dir: str, annotation_file_path: str, reverse: str = "n"):
    if reverse=="y":
        full_dataset_train = PDL1Dataset(input_dir, annotation_file_path, test_transform)
        full_dataset_test = PDL1Dataset(input_dir, annotation_file_path, train_transform)
    else:
        full_dataset_train = PDL1Dataset(input_dir, annotation_file_path, train_transform)
        full_dataset_test = PDL1Dataset(input_dir, annotation_file_path, test_transform)
    
    train_size: int = int(len(full_dataset_train)*0.8)
    test_size: int = len(full_dataset_train) - train_size

    train_data, test_indices = random_split(full_dataset_train, [train_size, test_size])

    test_data = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PDL1Dataset.collate_fn
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PDL1Dataset.collate_fn
    )
    return train_loader, test_loader

def get_lung_cancer_dataset(input_dir: str, reverse: str = "n"):
    if reverse=="y":
        full_dataset_train = LungCancerDataset(input_dir, test_transform)
        full_dataset_test = LungCancerDataset(input_dir, train_transform)
    else:
        full_dataset_train = LungCancerDataset(input_dir, train_transform)
        full_dataset_test = LungCancerDataset(input_dir, test_transform)
    
    train_size: int = int(len(full_dataset_train)*0.8)
    test_size: int = len(full_dataset_train) - train_size

    train_data, test_indices = random_split(full_dataset_train, [train_size, test_size])

    test_data = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def get_cifarten_dataset(reverse: str = "n"):
    if reverse=="y":
        train_data = datasets.CIFAR10("data", train=True, transform=test_transform, download=True)
        test_data = datasets.CIFAR10("data", transform=train_transform, train=False, download=True)
    else:
        train_data = datasets.CIFAR10("data", train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10("data", transform=test_transform, train=False, download=True)

    ## batchify data

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )
    return train_loader, test_loader

## LOAD MRI DATASET

def get_mri_dataset(input_folder: str, reverse: str = "n"):
    # inverse transform for measuting generalization
    if reverse=="y":
        full_dataset_train = MRIImageDataset(input_folder, transform=train_transform)
        full_dataset_test = MRIImageDataset(input_folder, transform=test_transform)
    else:
        full_dataset_train = MRIImageDataset(input_folder, transform=test_transform)
        full_dataset_test = MRIImageDataset(input_folder, transform=train_transform)

    train_size = int(0.8 * len(full_dataset_train))
    test_size = len(full_dataset_train) - train_size

    mri_train_dset, test_indices = random_split(full_dataset_train, [train_size, test_size])
    mri_test_dset = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    mri_train_loader = DataLoader(
        dataset=mri_train_dset,
        batch_size=args.batch_size,
        shuffle=True
    )

    mri_test_loader = DataLoader(
        dataset=mri_test_dset,
        batch_size=args.batch_size,
        shuffle=False
    )
    return mri_train_loader, mri_test_loader

## LOAD CIFAR10.1 DATASET

def get_cifar_tendotone_dataset(data_path: str, label_path: str):
    cifar10dot1 = CIFAR10dot1Dataset(data_path, label_path, transform=test_transform)
    cifar101_test_loader = DataLoader(
        dataset=cifar10dot1,
        batch_size=args.batch_size,
    shuffle=False
    )
    return cifar101_test_loader

def load_dataset(dataset_name: str, logger, input_folder: str = "", reverse: str = "n"):
    datasets = {}
    if dataset_name == "cifar10":
        logger.info("Loading CIFAR-10 dataset...")
        train_loader, test_loader = get_cifarten_dataset(reverse)
        datasets["train_loader"] = train_loader
        datasets["test_loader"] = test_loader
    elif dataset_name == "cifar10dot1":
        logger.info("Loading CIFAR-10.1 dataset...")
        train_loader, test_loader = get_cifar_tendotone_dataset(input_folder)
        datasets["train_loader"] = train_loader
        datasets["test_loader"] = test_loader
    elif dataset_name == "mri":
        logger.info("Loading MRI dataset...")
        train_loader, test_loader = get_mri_dataset(input_folder, reverse)
        datasets["train_loader"] = train_loader
        datasets["test_loader"] = test_loader
    elif dataset_name == "lung-cancer":
        logger.info("Loading lung cancer dataset...")
        train_loader, test_loader = get_lung_cancer_dataset(input_folder, reverse)
        datasets["train_loader"] = train_loader
        datasets["test_loader"] = test_loader
    elif dataset_name == "pdl1":
        logger.info("Loading PD-L1 dataset...")
        train_loader, test_loader = get_pdl1_dataset(input_folder, args.annotation_path, reverse)
        datasets["train_loader"] = train_loader
        datasets["test_loader"] = test_loader
    elif dataset_name == "coco":
        logger.info("Loading COCO dataset...")
        train_loader, test_loader, occ_map, class_parent = load_coco_dataset(args.coco_train_folder, args.coco_train_annotation, reverse)
        datasets["train_loader"] = train_loader
        datasets["test_loader"] = test_loader
        datasets["occ_map"] = occ_map
        datasets["class_parent"] = class_parent
    else:
        logger.error("Provided dataset not found, throwing exception...")
        datasets["error"] = "Dataset has not been registered yet for JEPA model!"
    return datasets

def load_coco_dataset(input_dir: str, annotation_json: str, reverse):
    if reverse=="y":
        full_dataset_train = COCODataset(input_dir, annotation_json, test_transform)
        full_dataset_test = COCODataset(input_dir, annotation_json, train_transform)
    else:
        full_dataset_train = COCODataset(input_dir, annotation_json, train_transform)
        full_dataset_test = COCODataset(input_dir, annotation_json, test_transform)
    
    train_size: int = int(len(full_dataset_train)*0.8)
    test_size: int = len(full_dataset_train) - train_size

    train_data, test_indices = random_split(full_dataset_train, [train_size, test_size])

    test_data = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=COCODataset.collate_fn
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=COCODataset.collate_fn
    )
    
    return train_loader, test_loader, full_dataset_train.class_occ_map, full_dataset_train.class_parent