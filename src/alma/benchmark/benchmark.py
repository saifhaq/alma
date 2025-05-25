import logging
import time
from typing import Any, Dict, Union, Optional

from alma.benchmark.metrics import TorchModuleMetrics
import torch
import torch._dynamo
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Pipeline

from ..conversions.conversion_options import ConversionOption
from ..conversions.select import select_forward_call_function
from ..dataloader.create import create_single_tensor_dataloader
from ..utils.data import get_sample_data
from ..utils.multiprocessing import benchmark_error_handler, init_lazy_model
from .benchmark_config import BenchmarkConfig
from .log import log_results
from .warmup import warmup
from .metrics import TextGenerationPipelineMetrics, TorchModuleMetrics

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# @benchmark_error_handler
def benchmark(
    device: torch.device,
    model: Any,
    config: BenchmarkConfig,
    conversion: ConversionOption,
    data: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
) -> Dict[str, Union[torch.device, float, int, str, torch.dtype]]:
    """
    Benchmark the model using the given data loader. This function will benchmark the model using the
    given conversion method.

    NOTE: We wrap this function in an error handler to catch any exceptions that occur during the
    benchmarking process. This will allow us to pass the error and traceback back through any
    multiprocessing handler.

    Inputs:
    - device (torch.device): The device we are targetting.
    - model (Union[torch.nn.Module, Callable]): The model to benchmark, or callable which returns
    the model (used for memory efficiency when using multi-processing, as a means of creating
    isolated test environments).
    - config (BenchmarkConfig): The configuration for the benchmarking.
    - conversion (ConversionOption): The conversion method to benchmark.
    - data (Optional[torch.Tensor]): The input data to benchmark the model on.
    - data_loader (Optional[DataLoader]): The DataLoader to get samples of data from.

    Outputs:
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    - total_time (float): The total time taken for inference.
    - total_samples (int): The total number of samples benchmarked.
    - throughput (float): The throughput of the model.
    """
    # If the model is a LazyLoad instance, load it to initialize the model
    print("Loaded", model.is_loaded())
    model = init_lazy_model(model, device)

    # Get the number of samples to benchmark
    n_samples = config.n_samples
    batch_size = config.batch_size

    # Creates a dataloader with random data, of the same size as the input data sample
    # If the data_loader has been provided by the user, we use that one
    if not isinstance(data_loader, DataLoader):
        data_loader = create_single_tensor_dataloader(
            tensor_size=data.size(),
            num_tensors=n_samples,
            random_type="normal",
            random_params={"mean": 0.0, "std": 2.0},
            batch_size=batch_size,
            dtype=conversion.data_dtype,
        )
    else:
        assert data_loader.batch_size == config.batch_size, f"The `data_loader.batch_size` ({data_loader.batch_size}) does not match the `config.batch_size` ({config.batch_size})"
        assert data_loader.sampler.total_samples == config.n_samples, f"The `data_loader.sampler.total_samples` ({data_loader.sampler.total_samples}) does not match the `config.n_samples` ({config.n_samples})"

    # Get sample of data from dataloader. This overwrites the data tensor provided by the user
    # We also check it matches the config dtype, if it is a Tensor
    data = get_sample_data(data_loader, device, conversion)

    # Get the forward call of the model, which we will benchmark. We also return the device we will
    # benchmark on, since some conversions are only supported for certain devices, e.g.
    # PyTorch native quantized conversions requires CPU
    forward_call = select_forward_call_function(model, conversion.mode, data, device)

    # Clear all caches, etc.
    torch._dynamo.reset()

    # Warmup
    warmup(forward_call, data_loader, warmup_iters=3, device=device)

    # Benchmarking loop - only process full batches
    if isinstance(model, torch.nn.Module):
        result = torch_module_benchmark(forward_call, config, conversion, data_loader, device)
    elif isinstance(model, Pipeline):
        result = hf_pipeline_benchmark(model.tokenizer, forward_call, config, conversion, data_loader, device)

    if logger.root.level <= logging.DEBUG:
        log_results(result)

    return result


def torch_module_benchmark(
    forward_call: callable,
    config: BenchmarkConfig,
    conversion: ConversionOption,
    data_loader: DataLoader,
    device: torch.device,
) -> TorchModuleMetrics:

    # Get the number of samples to benchmark
    n_samples = config.n_samples
    total_samples = 0
    n_batches = (n_samples + config.batch_size - 1) // config.batch_size

    # Setup CUDA events for more accurate GPU timing if available
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.perf_counter()

    with torch.no_grad():
        for i, data in enumerate(
            tqdm(
                data_loader,
                desc=f"Benchmarking {conversion.mode} on {device}",
                total=n_batches,
            )
        ):
            # Transfer data to device
            data = data.to(device, non_blocking=config.non_blocking)

            # Run forward pass without per-batch timing
            _ = forward_call(data)

            total_samples += data.size(0)

            if total_samples > n_samples:
                break

    # End timing and synchronize if needed
    if device.type == "cuda":
        end_event.record()
        end_event.synchronize()
        total_elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert ms to seconds
    else:
        end_time = time.perf_counter()
        total_elapsed_time = end_time - start_time

    throughput = total_samples / total_elapsed_time if total_elapsed_time > 0 else 0

    result = TorchModuleMetrics(
        device=device,
        total_elapsed_time=total_elapsed_time,
        total_samples=total_samples,
        batch_size=config.batch_size,
        throughput=throughput,
        status="success",
        data_dtype=conversion.data_dtype,
    )
    return result


def hf_pipeline_benchmark(
    tokenizer: any,
    forward_call: callable,
    config: BenchmarkConfig,
    conversion: ConversionOption,
    data_loader: DataLoader,
    device: torch.device,
) -> TextGenerationPipelineMetrics:
    # Get the number of samples to benchmark
    n_prompts = config.n_samples
    total_prompts = 0
    total_input_tokens = 0
    total_output_tokens = 0
    n_batches = (n_prompts + config.batch_size - 1) // config.batch_size

    # Setup CUDA events for more accurate GPU timing if available
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.perf_counter()

    with torch.no_grad():
        for i, data in enumerate(
            tqdm(
                data_loader,
                desc=f"Benchmarking {conversion.mode} on {device}",
                total=n_batches,
            )
        ):
            # Run forward pass without per-batch timing
            output = forward_call(data)

            for input_, output_ in zip(data, output):
                input_token_len = len(tokenizer(input_)["input_ids"])
                input_str_len = len(input_)
                total_input_tokens += input_token_len
                for gen in output_:
                    # for elem in gen:
                    generated_tokens = tokenizer(gen["generated_text"][input_str_len:])["input_ids"]
                    total_output_tokens += len(generated_tokens)

            total_prompts += len(data)
            if total_prompts > n_prompts:
                break

    # End timing and synchronize if needed
    if device.type == "cuda":
        end_event.record()
        end_event.synchronize()
        total_elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert ms to seconds
    else:
        end_time = time.perf_counter()
        total_elapsed_time = end_time - start_time

    output_throughput = total_output_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
    input_throughput = total_input_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
    request_rate = total_prompts / total_elapsed_time if total_elapsed_time > 0 else 0

    result = TextGenerationPipelineMetrics(
        device=device,
        total_elapsed_time=total_elapsed_time,
        total_prompts=total_prompts,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        batch_size=config.batch_size,
        output_throughput=output_throughput,
        input_throughput=input_throughput,
        request_rate=request_rate,
        status="success",
        data_dtype=conversion.data_dtype,
    )
    return result
