import os
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class BenchmarkCustomImageDataset(Dataset):
    def __init__(self, img_dir: str, transform: transforms.Compose = None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = []
        self._gather_images(img_dir)

    def _gather_images(self, directory: str) -> None:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    self.img_paths.append(os.path.join(root, file))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.img_paths[idx % len(self.img_paths)]  # Circular indexing
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning 0 as label since it's not used for benchmarking


class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        data = pd.read_csv(annotations_file)
        self.img_labels = data["target"]
        self.imgs = data["file"]
        self.img_dir = img_dir
        self.transform = transform
        # self.target_transform = target_transform  # is this needed?

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs.iloc[idx])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path
