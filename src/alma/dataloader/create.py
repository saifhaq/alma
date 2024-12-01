from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .dataloader import SingleTensorDataset


def create_single_tensor_dataloader(
    tensor_size: Union[int, Tuple[int, ...]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    tensor: Optional[torch.Tensor] = None,
    num_tensors: int = 100,
    random_type: str = "normal",
    random_params: Optional[dict] = None,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Creates a DataLoader that serves either a single tensor or randomly generated tensors.

    Args:
        tensor_size: Size of the tensor(s) to serve
        batch_size: Size of each batch
        shuffle: Whether to shuffle the indices
        num_workers: Number of worker processes
        tensor: Optional specific tensor to serve
        num_tensors: Number of random tensors to generate if no specific tensor provided
        random_type: Type of random tensors to generate ('normal', 'uniform', 'bernoulli')
        random_params: Parameters for random tensor generation
        **dataloader_kwargs: Additional arguments to pass to DataLoader

    Returns:
        DataLoader: A PyTorch DataLoader that serves the tensor(s) repeatedly

    Example:
        >>> # Create a dataloader with random normal tensors
        >>> dataloader = create_single_tensor_dataloader(
        ...     tensor_size=(10, 5),
        ...     num_tensors=50,
        ...     random_type='normal',
        ...     random_params={'mean': 0.0, 'std': 1.0}
        ... )

        >>> # Create a dataloader with a specific tensor
        >>> specific_tensor = torch.randn(10, 5)
        >>> dataloader = create_single_tensor_dataloader(
        ...     tensor_size=(10, 5),
        ...     tensor=specific_tensor
        ... )
    """
    dataset: Dataset = SingleTensorDataset(
        tensor_size=tensor_size,
        num_tensors=num_tensors,
        tensor=tensor,
        random_type=random_type,
        random_params=random_params,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **dataloader_kwargs,
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Using a specific tensor
    specific_tensor = torch.randn(10, 5)
    dataloader1 = create_single_tensor_dataloader(
        tensor_size=(10, 5), tensor=specific_tensor, batch_size=4
    )

    # Example 2: Generate random normal tensors
    dataloader2 = create_single_tensor_dataloader(
        tensor_size=(10, 5),
        num_tensors=50,
        random_type="normal",
        random_params={"mean": 0.0, "std": 2.0},
        batch_size=4,
    )

    # Example 3: Generate random uniform tensors
    dataloader3 = create_single_tensor_dataloader(
        tensor_size=(10, 5),
        num_tensors=50,
        random_type="uniform",
        random_params={"low": -1.0, "high": 1.0},
        batch_size=4,
    )

    # Print first few batches from each dataloader
    for name, dataloader in [
        ("Specific Tensor", dataloader1),
        ("Random Normal", dataloader2),
        ("Random Uniform", dataloader3),
    ]:
        print(f"\n{name} Dataloader:")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i + 1}, shape: {batch.shape}")
            if i >= 2:  # Just print first 3 batches
                break
