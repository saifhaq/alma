import torch


class DeviceCollator:
    """
    A collate function class that moves batched data to a specified device.

    Attributes:
        device (torch.device): The target device for the batched data
    """

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, batch):
        # Default collation
        data = torch.utils.data.default_collate(batch)
        # Move to device during collation
        return tuple(
            x.to(self.device) if isinstance(x, torch.Tensor) else x for x in data
        )

