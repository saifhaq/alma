import logging

import torch
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.utils.data import DataLoader

from arguments.infer_args import parse_infer_args
from conversions.compile import get_compiled_model
from data.datasets import InferenceDataset
from data.transforms import InferenceTransform
from data.utils import gather_image_paths
from utils.ipdb_hook import ipdb_sys_excepthook
from utils.load_model import load_model
from utils.setup_logging import setup_logging
from utils.times import inference_time_benchmarking, log_benchmark_times

# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


ipdb_sys_excepthook()


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
    args, device = parse_infer_args()

    # Load the model
    times = {}
    model, times = load_model(args.model, device, times, logging=logging)

    # Gather image paths
    logging.info(f"Gathering image paths from {args.target}")
    image_paths, times = gather_image_paths(args.target, times)

    # Create a DataLoader for the inference dataset
    dataset = InferenceDataset(image_paths, transform=InferenceTransform)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    logging.info("Starting eager inference...")
    digit_counts, times = inference_time_benchmarking(data_loader, model, device, times)
    log_benchmark_times(logging, times)

    logging.info("Starting compiled inference...")

    compiled_model = get_compiled_model(model, data_loader, device, logging)
    digit_counts, times = inference_time_benchmarking(
        data_loader, compiled_model, device, times
    )
    log_benchmark_times(logging, times)

    logging.info("Inference complete.")

    for digit, count in enumerate(digit_counts):
        logging.info(f"Digit {digit}: {count}")


if __name__ == "__main__":
    main()
