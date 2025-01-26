import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class Caltech101Dataset(Dataset):
    """
    Base dataset class - accommodates both standard and contrastive training paradigms
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 split_file: Optional[str] = None,
                 mode: str = 'standard',
                 corruption_types: Optional[List[str]] = None,
                 corruption_levels: Optional[List[int]] = None,
                 transform=None):
        """
        Args:
            root_dir: Path to the clean/noisy separated caltech-101 directory
            split: One of 'train', 'val', or 'test'
            split_file: Path to JSON file containing train/val/test splits - created if kept empty
            mode: One of 'standard' or 'contrastive'
            corruption_types: List of corruption types to include (None = all)
            corruption_levels: List of corruption levels to include (None = all)
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.mode = mode
        self.transform = transform

        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.splits = json.load(f)
        else:
            self.splits = self._create_splits()
            if split_file:
                with open(split_file, 'w') as f:
                    json.dump(self.splits, f)
                print(f"Splits file saved to: {split_file}")

        self.classes = sorted([d.name for d in (self.root_dir / 'clean').iterdir()
                             if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        all_corruptions = [d.name for d in (self.root_dir / 'corrupted').iterdir()
                          if d.is_dir()]
        self.corruption_types = corruption_types or all_corruptions
        self.corruption_levels = corruption_levels or list(range(1, 6))

        # Create image list for the specified split
        self.samples = self._create_sample_list()

    def _create_splits(self, val_size=0.1, test_size=0.1) -> Dict:
        clean_images = []
        for class_dir in (self.root_dir / 'clean').iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.jpeg'):
                    clean_images.append(str(img_path.relative_to(self.root_dir / 'clean')))

        # First split into train and temp
        train_imgs, temp_imgs = train_test_split(
            clean_images, test_size=(val_size + test_size), random_state=42
        )

        # Split temp into val and test
        relative_val_size = val_size / (val_size + test_size)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5, random_state=42
        )

        return {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

    def _create_sample_list(self) -> List[Dict]:
        """Create list of samples for the current split"""
        samples = []
        split_files = self.splits[self.split]

        for relative_path in split_files:
            class_name = Path(relative_path).parent.name
            image_id = Path(relative_path).stem

            # Add clean image
            clean_path = self.root_dir / 'clean' / relative_path
            samples.append({
                'image_path': str(clean_path),
                'class': class_name,
                'class_idx': self.class_to_idx[class_name],
                'is_clean': True,
                'corruption_type': None,
                'corruption_level': None,
                'clean_path': str(clean_path)
            })

            # Add corrupted versions
            for corruption in self.corruption_types:
                for level in self.corruption_levels:
                    corrupted_path = (self.root_dir / 'corrupted' / corruption /
                                    str(level) / relative_path)
                    if corrupted_path.exists():
                        samples.append({
                            'image_path': str(corrupted_path),
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name],
                            'is_clean': False,
                            'corruption_type': corruption,
                            'corruption_level': level,
                            'clean_path': str(clean_path)
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int],
                                            Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Returns:
            For mode='standard': (image, class_idx)
            For mode='contrastive': (corrupted_image, clean_image, class_idx)
        """
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'standard':
            return image, sample['class_idx']
        else:  # contrastive mode
            clean_image = Image.open(sample['clean_path']).convert('RGB')
            if self.transform:
                clean_image = self.transform(clean_image)
            return image, clean_image, sample['class_idx']

def get_dataloaders(root_dir: str,
                    mode: str = 'standard',
                    num_workers: int = 2,
                    split_file: Optional[str] = None,
                    batch_size: int = 32,
                    **dataset_kwargs) -> Dict[str, DataLoader]:
    """
    Create dataloaders for train, validation and test sets

    Args:
        root_dir: Path to dataset
        mode: 'standard' or 'contrastive'
        num_workers: Number of workers for dataloaders
        split_file: Path to split file
        **dataset_kwargs: Additional arguments to pass to Caltech101Dataset

    Returns:
        Dictionary containing train, val, and test dataloaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = Caltech101Dataset(
            root_dir=root_dir,
            split=split,
            split_file=split_file,
            mode=mode,
            transform=transform,
            **dataset_kwargs
        )

    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    return dataloaders

def set_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    print(f"Testing Dataloader Splits")
    # Set all seeds
    SEED = 42
    set_seeds(SEED)
    
    os.makedirs('splits', exist_ok=True)
    split_file = f'../splits/{SEED}.json'

    standard_loaders = get_dataloaders(
        root_dir="../caltech-101",
        mode='standard', 
        split_file=split_file,
        batch_size = 1024
    )

    contrastive_loaders = get_dataloaders(
        root_dir="../caltech-101",
        mode='contrastive',
        split_file=split_file,
        batch_size = 1024
    )