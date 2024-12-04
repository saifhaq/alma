def check_consistent_batch_size(
    conversion_method: str, n_samples: int, batch_size: int
) -> None:
    """
    Check that the batch size will always be consistent, which is a requirement for certain
    conversion methods.

    Inputs:
    - conversion_method (str): the conversion method being evaluated.
    - n_samples (int): the number of samples to use during benchmarking. This may have to be a perfect
        multiple of the batch size, depending on the conversion method.
    - batch_size (int): the batch size to use during benchmarking.
    """
    if "COMPILE" in conversion_method:
        if n_samples % batch_size != 0:
            error_msg = f"""n_samples must be a multiple of batch_size for compilation based methods, 
otherwise the last batch will fail due to having a different shape. n-samples was {n_samples}, and 
batch size was {batch_size} """
            raise ValueError(error_msg)
