from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class CircularSampler(Sampler):
    """
    A sampler that provides an infinite stream of indices.
    """

    def __init__(self, data_source: Dataset):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from torch.randperm(len(self.data_source)).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class CircularDataLoader(DataLoader):
    """
    A DataLoader that uses the CircularSampler to provide an infinite stream of data.
    """

    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False, **kwargs
    ):
        sampler = CircularSampler(dataset)
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)
