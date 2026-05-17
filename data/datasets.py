import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from data.label_mapping import cxrigen_label_to_chexpert


class CheXpertDataset(Dataset):
    """Real CheXpert dataset with the U-Ones uncertainty policy (-1 -> 1, NaN -> 0)."""

    def __init__(self, csv_path, data_root, transform=None):
        df = pd.read_csv(csv_path)
        df = df.fillna(0).replace(-1, 1)
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row['Path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.FloatTensor(row.iloc[5:].values.astype(float))
        return image, labels


class SyntheticDataset(Dataset):
    """CXR-IRGen synthetic dataset with optional filename filtering."""

    def __init__(self, synthetic_dir, metadata_csv, transform=None, allowed_filenames=None):
        df = pd.read_csv(metadata_csv)
        if allowed_filenames is not None:
            df = df[df['filename'].isin(set(allowed_filenames))]
        self.df = df.reset_index(drop=True)
        self.synthetic_dir = synthetic_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.synthetic_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        irgen_labels = ast.literal_eval(row['labels'])
        chexpert_labels = cxrigen_label_to_chexpert(irgen_labels)
        return image, torch.FloatTensor(chexpert_labels)


class MixedDataset(Dataset):
    """Concatenation of a real and a synthetic dataset for joint training."""

    def __init__(self, real_dataset, synthetic_dataset):
        self.real = real_dataset
        self.synthetic = synthetic_dataset
        self.real_len = len(real_dataset)

    def __len__(self):
        return self.real_len + len(self.synthetic)

    def __getitem__(self, idx):
        if idx < self.real_len:
            return self.real[idx]
        return self.synthetic[idx - self.real_len]
