from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class SingleTensorDataset(Dataset):
    """
    A PyTorch Dataset that serves either a single tensor or randomly generated tensors repeatedly.

    Inputs:
    - tensor_size (Union[int, Tuple[int, ...]]): Size of the tensor to serve.
                                                Can be a single integer or a tuple of dimensions.
    - num_tensors (int, optional): Number of random tensors to generate. Only used if tensor is None.
    - tensor (torch.Tensor, optional): A specific tensor to serve. If provided, num_tensors is ignored.
    - random_type (str, optional): Type of random tensor to generate ('normal', 'uniform', 'bernoulli').
                                 Only used if tensor is None.
    - random_params (dict, optional): Parameters for random tensor generation.
            - For 'normal': {'mean': float, 'std': float}
            - For 'uniform': {'low': float, 'high': float}
            - For 'bernoulli': {'p': float}
    """

    def __init__(
        self,
        tensor_size: Union[int, Tuple[int, ...]],
        num_tensors: int = 100,
        tensor: Optional[torch.Tensor] = None,
        random_type: str = "normal",
        random_params: Optional[dict] = None,
    ):
        self.tensor_size = (
            tensor_size if isinstance(tensor_size, tuple) else (tensor_size,)
        )

        if tensor is not None:
            # Use the provided tensor
            self.tensors = [tensor]
            self.length = 1
        else:
            # Generate random tensors
            self.length = num_tensors
            self.tensors = self._generate_random_tensors(
                random_type, random_params or {}, num_tensors
            )

    def _generate_random_tensors(
        self, random_type: str, params: dict, num_tensors: int
    ) -> List[torch.Tensor]:
        """Generate a list of random tensors based on specified parameters."""
        tensors = []

        for _ in range(num_tensors):
            if random_type == "normal":
                mean = params.get("mean", 0.0)
                std = params.get("std", 1.0)
                tensor = torch.normal(mean, std, size=self.tensor_size)

            elif random_type == "uniform":
                low = params.get("low", 0.0)
                high = params.get("high", 1.0)
                tensor = torch.uniform_(torch.zeros(self.tensor_size), low, high)

            elif random_type == "bernoulli":
                p = params.get("p", 0.5)
                tensor = torch.bernoulli(torch.full(self.tensor_size, p))

            else:
                raise ValueError(
                    f"Unsupported random_type: {random_type}. "
                    "Use 'normal', 'uniform', or 'bernoulli'."
                )

            tensors.append(tensor)

        return tensors

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return the tensor at the specified index, and the index itself
        return self.tensors[idx], idx
