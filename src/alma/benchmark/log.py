import logging
from typing import Dict


def log_results(
    results: Dict[str, float],
) -> None:
    """
    Logs the benchmarking results.

    Inputs:
    - results (Dict[str, float]): The results of the benchmarking. This contains the following:
        - total_elapsed_time (float): The total elapsed time for the benchmark.
        - total_inf_time (float): The total time taken for inference.
        - total_samples (int): The total number of samples benchmarked.
        - throughput (float): The throughput of the model.

    Outputs:
    None
    """
    logging.info(f"Total elapsed time: {results['total_elapsed_time']:.4f} seconds")
    logging.info(
        f"Total inference time (model only): {results['total_inf_time']:.4f} seconds"
    )
    logging.info(f"Total samples: {results['total_samples']}")
    logging.info(f"Throughput: {results['throughput']:.2f} samples/second")
