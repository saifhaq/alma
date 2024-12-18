from typing import Iterator

import torch
from torch.utils.data import Dataset


class CircularSampler(torch.utils.data.Sampler):
    """A sampler that provides an infinite stream of indices."""

    def __init__(self, data_source: Dataset):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        while True:  # Infinite loop
            yield from torch.randperm(len(self.data_source)).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class CircularDataLoader(torch.utils.data.DataLoader):
    """A DataLoader that uses CircularSampler for infinite data streams."""

    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False, **kwargs
    ):
        sampler = CircularSampler(dataset)
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)


class DummyDataset(Dataset):
    def __init__(self, size: int = 100):
        self.data = torch.randn(size, 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    dataset = DummyDataset(size=100)
    data_loader = CircularDataLoader(dataset, batch_size=8, shuffle=True)

    print("DataLoader created successfully.")

    # Retrieve a fixed number of batches (e.g., 3)
    max_batches = 3
    for i, batch in enumerate(data_loader):
        print(f"Batch {i + 1}:", batch)
        if i + 1 >= max_batches:
            break  # Stop after 3 batches
