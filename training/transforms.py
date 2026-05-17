import torchvision.transforms as transforms

from config import IMAGENET_MEAN, IMAGENET_STD, IMG_CROP

_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def get_train_transforms():
    """Standard training transforms (random resized crop + horizontal flip)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_CROP),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize,
    ])


def get_val_transforms():
    """Deterministic validation transforms (fixed resize)."""
    return transforms.Compose([
        transforms.Resize((IMG_CROP, IMG_CROP)),
        transforms.ToTensor(),
        _normalize,
    ])


def get_heavy_train_transforms():
    """Heavy traditional augmentation pipeline used in WP6 cell G."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_CROP, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        _normalize,
    ])
