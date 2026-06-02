import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PotholeDataset(Dataset):
    """Dataset class for pothole segmentation"""
    
    def __init__(self, image_dir, mask_dir, image_size=512, augment=False, mode='train'):
        """
        Args:
            image_dir (str): Path to directory with images
            mask_dir (str): Path to directory with segmentation masks
            image_size (int): Size to resize images to
            augment (bool): Whether to apply augmentation
            mode (str): 'train', 'val', or 'test'
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment and mode == 'train'
        
        # Get list of images
        self.images = sorted(self.image_dir.glob('*.jpg')) + sorted(self.image_dir.glob('*.png'))
        
        # Define augmentation
        if self.augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussBlur(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], is_check_shapes=False)
        else:
            self.aug = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], is_check_shapes=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # If mask doesn't exist, create empty mask
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        
        # Apply augmentation
        if self.augment:
            augmented = self.aug(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            augmented = self.aug(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        return {
            'image': img,
            'mask': mask.float(),
            'path': str(img_path)
        }


class DataLoader:
    """Data loader wrapper"""
    
    def __init__(self, image_dir, mask_dir, batch_size=16, image_size=512, 
                 num_workers=4, augment=True, mode='train'):
        self.dataset = PotholeDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            augment=augment,
            mode=mode
        )
        
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        return iter(self.loader)
