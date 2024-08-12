import argparse
import logging
import os
import time
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

from train import Net


def setup_logging(log_file: str = None) -> None:
    """
    Sets up logging to print to console and optionally to a file.

    Args:
        log_file (str): Path to a log file. If None, logs will not be saved to a file.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ],
    )


class CustomImageDataset(Dataset):
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


class CircularSampler(Sampler):
    def __init__(self, data_source: Dataset):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from torch.randperm(len(self.data_source)).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class CircularDataLoader(DataLoader):
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False, **kwargs
    ):
        sampler = CircularSampler(dataset)
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if model_path.endswith(".pt"):
        try:
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location=device)
            model = Net()  # Define your model architecture (Net)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
    else:
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.eval()
    return model


def benchmark_model(
    model: torch.nn.Module, device: torch.device, data_loader: DataLoader, n_images: int
) -> None:
    total_time = 0.0
    total_images = 0
    num_batches = 0

    start_time = time.time()  # Start timing for the entire process

    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Benchmarking"):
            if total_images >= n_images:
                break

            data = data.to(device)
            batch_start_time = time.time()
            output = model(data)
            batch_end_time = time.time()

            batch_size = min(data.size(0), n_images - total_images)
            total_time += batch_end_time - batch_start_time
            total_images += batch_size
            num_batches += 1

            if total_images >= n_images:
                break

    end_time = time.time()  # End timing for the entire process

    total_elapsed_time = end_time - start_time
    throughput = total_images / total_elapsed_time if total_elapsed_time > 0 else 0
    logging.info(f"Total elapsed time: {total_elapsed_time:.4f} seconds")
    logging.info(f"Total inference time (model only): {total_time:.4f} seconds")
    logging.info(f"Total images: {total_images}")
    logging.info(f"Throughput: {throughput:.2f} images/second")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Models")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model file (e.g., mnist_cnn_quantized.pt, mnist_cnn_scripted.pt, mnist_cnn.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing images for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        metavar="N",
        help="input batch size for benchmarking (default: 30)",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=5000,
        help="Total number of images to process (default: 300)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA acceleration",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables MPS acceleration",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Ensure all images are resized to 28x28
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = CustomImageDataset(img_dir=args.data_dir, transform=transform)

    data_loader = CircularDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model_path, device)

    benchmark_model(model, device, data_loader, args.n_images)


if __name__ == "__main__":
    setup_logging()
    main()
