import argparse
import concurrent.futures
import logging
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

from ipdb_hook import ipdb_sys_excepthook
from train import Net

ipdb_sys_excepthook()


def setup_logging(log_file=None):
    """
    Sets up logging to print to console and optionally to a file.

    Args:
        log_file (str): Path to a log file. If None, logs will not be saved to a file.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_level = logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


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


def gather_image_paths(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def process_batch(batch, model, device):
    data = batch.to(device)
    output = model(data)
    preds = output.argmax(dim=1, keepdim=True)
    return preds.cpu().numpy()


def compile_model(model: torch.nn.Module):
    import torch._dynamo

    torch._dynamo.reset()
    model_opt = torch.compile(model, mode="reduce-overhead")
    return model_opt

    # Step 2. quantization
    # backend developer will write their own Quantizer and expose methods to allow
    # users to express how they
    # want the model to be quantized
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    # or prepare_qat_pt2e for Quantization Aware Training
    m = prepare_pt2e(model_opt, quantizer)


def export_to_tensorRT(model: torch.nn.Module):
    # Placeholder
    raise NotImplementedError("This method is not implemented yet")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Digit Classification Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="path to the trained TorchScript model",
        default="mnist_cnn_scripted.pt",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        help="directory with images for inference",
        default="data_for_inference",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    logging.info(f"Loading model from {args.model}")
    model_load_start_time = time.time()
    if "scripted" in args.model:
        model = torch.jit.load(args.model).to(device)
    else:
        model = Net()
        model.load_state_dict(torch.load(args.model))
        model.to(device)
    model.eval()
    model_load_end_time = time.time()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),  # Resize images to 28x28
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    logging.info(f"Gathering image paths from {args.target}")
    image_paths_gather_start_time = time.time()
    image_paths = gather_image_paths(args.target)
    image_paths_gather_end_time = time.time()
    dataset = InferenceDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    digit_counts = [0] * 10

    def inference_benchmarking(
        data_loader, model, device
    ) -> Tuple[Dict[int, int], float]:
        inference_start_time = time.time()
        with torch.no_grad():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for batch, _ in tqdm(
                    data_loader, desc="Processing batches", unit="batch"
                ):
                    futures.append(executor.submit(process_batch, batch, model, device))
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Aggregating results",
                    unit="batch",
                ):
                    preds = future.result()
                    for pred in preds:
                        digit_counts[pred.item()] += 1
        inference_end_time = time.time()
        return digit_counts, inference_start_time, inference_end_time

    def calc_benchmark_times(
        model_load_start_time,
        model_load_end_time,
        image_paths_gather_start_time,
        image_paths_gather_end_time,
        inference_start_time,
        inference_end_time,
    ):
        total_execution_time = inference_end_time - model_load_start_time
        inference_time = inference_end_time - inference_start_time
        model_load_time = model_load_end_time - model_load_start_time
        image_paths_gather_time = (
            image_paths_gather_end_time - image_paths_gather_start_time
        )

        logging.info(f"Model loading time: {model_load_time:.2f} seconds")
        logging.info(
            f"Image paths gathering time: {image_paths_gather_time:.2f} seconds"
        )
        logging.info(f"Inference time: {inference_time:.2f} seconds")
        logging.info(f"Total execution time: {total_execution_time:.2f} seconds")
        return (
            total_execution_time,
            inference_time,
            model_load_time,
            image_paths_gather_time,
        )

    logging.info("Starting eager inference...")
    digit_counts, inference_start_time, inference_end_time = inference_benchmarking(
        data_loader, model, device
    )
    calc_benchmark_times(
        model_load_start_time,
        model_load_end_time,
        image_paths_gather_start_time,
        image_paths_gather_end_time,
        inference_start_time,
        inference_end_time,
    )

    logging.info("Starting compiled inference...")
    compiled_model = compile_model(model)
    digit_counts, inference_start_time, inference_end_time = inference_benchmarking(
        data_loader, compiled_model, device
    )
    calc_benchmark_times(
        model_load_start_time,
        model_load_end_time,
        image_paths_gather_start_time,
        image_paths_gather_end_time,
        inference_start_time,
        inference_end_time,
    )

    logging.info("Inference complete.")

    for digit, count in enumerate(digit_counts):
        logging.info(f"Digit {digit}: {count}")


if __name__ == "__main__":
    main()
