import time
from typing import Any, Callable, Dict

import torch
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def time_forward(
    forward_call: Callable,
    data_loader: torch.utils.data.DataLoader,
    n_samples: int,
    device: torch.device,
    conversion: str,
    warmup_iterations: int = 10,
) -> dict:
    """
    Time the forward pass of the model using the given dataloader. Internally determines whether to use
    CUDA timing or generic timing.

    Inputs:
    - forward_call (Callable): The forward call to time.
    - data_loader (DataLoader): The dataloader to get the data from.
    - n_samples (int): The number of samples to time.
    - device (torch.device): The device to run the model on.
    - conversion (str): The conversion method used.
    - warmup_iterations (int): The number of warmup iterations to perform.

    Outputs (Dict):
    - total_inf_time (float): The total inference time.
    - total_samples (int): The total number of samples timed.
    - num_batches (int): The number of batches timed.
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    """
    if len(data_loader) == 0:
        raise ValueError("Empty data loader provided")

    # if device.type in ["cudax", "XLA"]:
    #     print("Using CUDA timing module")
    result = time_cuda(
        forward_call,
        data_loader,
        n_samples,
        device,
        conversion,
        warmup_iterations=10,
    )
    # else:
    #     print("Using generic timing module")
    #     result = time_generic(
    #         forward_call,
    #         data_loader,
    #         n_samples,
    #         device,
    #         conversion,
    #         warmup_iterations=10,
    #     )
    return result


def time_cuda(
    forward_call: Callable,
    data_loader: torch.utils.data.DataLoader,
    n_samples: int,
    device: torch.device,
    conversion: str,
    warmup_iterations: int = 10,
) -> dict:
    """
    Time the forward pass of the model using the given dataloader. Specifically made for CUDA timing.

    Inputs:
    - forward_call (Callable): The forward call to time.
    - data_loader (DataLoader): The dataloader to get the data from.
    - n_samples (int): The number of samples to time.
    - device (torch.device): The device to run the model on.
    - conversion (str): The conversion method used.
    - warmup_iterations (int): The number of warmup iterations to perform.

    Outputs (Dict):
    - total_inf_time (float): The total inference time.
    - total_samples (int): The total number of samples timed.
    - num_batches (int): The number of batches timed.
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    """
    # Start timing for the entire process
    start_time = time.perf_counter()

    # Ensure CUDA is initialized
    # torch.cuda.synchronize()

    # Create dedicated CUDA stream for timing
    # stream = torch.cuda.Stream()

    # Do warmup iterations to get GPU to steady state
    with torch.no_grad():  #, torch.cuda.stream(stream):
        warmup_data = next(iter(data_loader))[0]
        print(f"Data device: {warmup_data.device}")
        for _ in range(warmup_iterations):
            _ = forward_call(warmup_data)


    # Initialize benchmark variables
    total_inf_time = 0.0
    total_samples = 0
    num_batches = 0
    batch_sizes = []
    expected_batch_size = warmup_data.size(0)

    start_events = []
    end_events = []

    # Main timing loop
    with torch.no_grad(): #, torch.cuda.stream(stream):
        for i, (data, _) in enumerate(
            tqdm(data_loader, desc=f"Benchmarking {conversion} on {device}")
        ):
            if total_samples >= n_samples:
                break

            # Create events for this batch
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Record timing for entire batch
            start_event.record()
            _ = forward_call(data)
            end_event.record()

            # Store events
            start_events.append(start_event)
            end_events.append(end_event)

            batch_sizes.append(data.size(0))
            total_samples += data.size(0)
            num_batches += 1

    # Make sure all operations are completed
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) / 1000 for s, e in zip(start_events, end_events)]
    total_inf_time = sum(times)

    # End timing for the entire process
    end_time = time.perf_counter()

    # Cleanup CUDA events
    torch.cuda.empty_cache()

    assert all(
        expected_batch_size == batch_size for batch_size in batch_sizes
    ), f"All batches should have the same size. Got batch sizes: {batch_sizes}"

    total_elapsed_time = (end_time - start_time) / 1000
    throughput = total_samples / total_inf_time if total_inf_time > 0 else 0
    result: Dict[str, Any] = {
        "device": device,
        "total_elapsed_time": total_elapsed_time,
        "total_inf_time": total_inf_time,
        "total_samples": total_samples,
        "batch_size": batch_sizes[0],
        "throughput": throughput,
    }
    return result


def time_generic(
    forward_call: Callable,
    data_loader: torch.utils.data.DataLoader,
    n_samples: int,
    device: torch.device,
    conversion: str,
    warmup_iterations: int = 10,
) -> dict:
    """
    Time the forward pass of the model using the given dataloader. Specifically made for CPU timing.

    Inputs:
    - forward_call (Callable): The forward call to time.
    - data_loader (DataLoader): The dataloader to get the data from.
    - n_samples (int): The number of samples to time.
    - device (torch.device): The device to run the model on.
    - conversion (str): The conversion method used.
    - warmup_iterations (int): The number of warmup iterations to perform.

    Outputs (Dict):
    - total_inf_time (float): The total inference time in milliseconds.
    - total_samples (int): The total number of samples timed.
    - num_batches (int): The number of batches timed.
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    """
    # Start timing for the entire process
    start_time = time.perf_counter()

    # Initialize device-specific synchronization
    def sync_device():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type != "cpu":
            torch.cuda.synchronize()

    # Do warmup iterations to get device to steady state
    with torch.no_grad():
        warmup_data = next(iter(data_loader))[0].to(device)
        for _ in range(warmup_iterations):
            _ = forward_call(warmup_data)
            sync_device()

    # Initialize benchmark variables
    total_inf_time = 0.0
    total_samples = 0
    num_batches = 0
    batch_sizes = []

    # Pre-calculate number of steps
    steps = min(
        len(data_loader),
        (n_samples + data_loader.batch_size - 1) // data_loader.batch_size,
    )

    # Create timing arrays
    start_times = [0.0] * steps
    end_times = [0.0] * steps

    # Main timing loop
    with torch.no_grad():
        for i, (data, _) in enumerate(
            tqdm(data_loader, desc=f"Benchmarking {conversion} on {device}")
        ):
            if total_samples >= n_samples:
                break

            data = data.to(device)
            # sync_device()  # Ensure previous operations are complete

            # Record timing for entire batch
            start_times[i] = time.perf_counter()
            _ = forward_call(data)
            # sync_device()  # Ensure forward pass is complete
            end_times[i] = time.perf_counter()

            batch_sizes.append(data.size(0))
            total_samples += data.size(0)
            num_batches += 1

        # Calculate total inference time in seconds
        times = [
            (end - start)
            for start, end in zip(start_times[:num_batches], end_times[:num_batches])
        ]
        total_inf_time = sum(times)

    # End timing for the entire process
    end_time = time.perf_counter()

    assert all(
        batch_sizes[0] == batch_size for batch_size in batch_sizes
    ), f"All batches should have the same size. Got batch sizes: {batch_sizes}"

    total_elapsed_time = end_time - start_time
    throughput = total_samples / total_inf_time if total_elapsed_time > 0 else 0

    result: Dict[str, Any] = {
        "device": device,
        "total_elapsed_time": total_elapsed_time,
        "total_inf_time": total_inf_time,
        "total_samples": total_samples,
        "batch_size": batch_sizes[0],
        "throughput": throughput,
    }

    return result
