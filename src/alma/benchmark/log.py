import logging

def log_results(
    logger: logging.Logger,
    total_elapsed_time: float,
    total_time: float,
    total_samples: float,
    throughput: float,
) -> None:
    """
    Logs the benchmarking results.

    Inputs:
    - logger (logging.Logger): The logger to use for logging.
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    - total_time (float): The total time taken for inference.
    - total_samples (int): The total number of samples benchmarked.
    - throughput (float): The throughput of the model.

    Outputs:
    None
    """
    logger.info(f"Total elapsed time: {total_elapsed_time:.4f} seconds")
    logger.info(f"Total inference time (model only): {total_time:.4f} seconds")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Throughput: {throughput:.2f} samples/second")


