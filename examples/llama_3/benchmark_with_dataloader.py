import logging
from typing import Any, Dict

import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.setup_logging import setup_logging

from data import CircularSampler, PromptDataset

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


# It is a lot more memory efficienct, if multi-processing is enabled, to create the model in a
# callable function, which can be called later to create the model.
# This allows us to initialise the model in each child process, rather than the parent
# process. This is because the model is not pickled and sent to the child process (which would
# require the program to sotre the model in memory twice), but rather created in the child
# process. This is especially important if the model is large and two instances would not fit
# on device.
def pipe_call() -> callable:
    """
    A callable that returns the model to be benchmarked. This allows us to initialise the model
    at a later date, which is useful when using multiprocessing (in turn used to isolate each method
    in its own process, keeping them from contaminating the global torch state). By initialising
    the model inside the child process, we avoid having two instances of the model in memory at
    once.

    NOTE: THIS HAS TO BE DEFINED AT THE MODULE LEVEL, NOT NESTED INSIDE ANY FUNCTION. This is so
    that it is pickle-able, necessary for it to be passed to multi-processing.

    For pipelines, for any kwargs we want to pass the model (e.g. `max_new_tokens`), these should
    be defined at pipeline initialization as in the below example.
    """
    # Initialise HF transformers pipeline object, which will be our high-level model wrapper
    # we can pass to alma
    # REQUIRES HUGGINGFACE_TOKEN TO BE AN ENVIRONMENTAL VARIABLE AND TO HAVE ACCEPTED THE META LICENSE
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            """`HF_TOKEN` should be provided as an environmental variable, and one 
should accept the Meta terms and conditions to download the LLama 3 8B Instruct model from HuggingFace"""
        )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=None,  # See issue: https://github.com/huggingface/transformers/issues/19358
        max_new_tokens=100,  # Maximum tokens we allow the LLM to generate
        min_new_tokens=99,  # Minimum tokens we allow the LLM to generate
        num_return_sequences=1,  # How many sequences we want the model to return per prompt
    )
    return pipe


def main() -> None:
    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="INFO")

    args, conversions = parse_benchmark_args()

    # Create dataset and circular dataloader. We add a random string input to reduce cache hits on
    # the prompts.
    dataset = PromptDataset(include_random_prefix=True)
    sampler = CircularSampler(dataset, args.n_samples)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Set the device one wants to benchmark on
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialise HF transformers pipeline object, which will be our high-level model wrapper
    # we can pass to alma
    # REQUIRES HUGGINGFACE_TOKEN TO BE AN ENVIRONMENTAL VARIABLE AND TO HAVE ACCEPTED THE META LICENSE
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            """`HF_TOKEN` should be provided as an environmental variable, and one 
should accept the Meta terms and conditions to download the LLama 3 8B Instruct model from HuggingFace"""
        )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    @lazyload
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=None,  # See issue: https://github.com/huggingface/transformers/issues/19358
        max_new_tokens=100,  # Maximum tokens we allow the LLM to generate
        min_new_tokens=99,  # Minimum tokens we allow the LLM to generate
        num_return_sequences=1,  # How many sequences we want the model to return per prompt
    )

    # Configuration for the benchmarking
    config = BenchmarkConfig(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        multiprocessing=False,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        fail_on_error=True,  # If False, we fail gracefully and keep testing other methods
        non_blocking=False,  # If True, we don't block the main thread when transferring data from host to device
    )

    # Hard-code a list of options. These can be provided as a list of strings, or a list of ConversionOption objects
    conversions = [
        "EAGER",
        "JIT_TRACE",
        "COMPILE_INDUCTOR_DEFAULT",
        "COMPILE_OPENXLA",
        "COMPILE_INDUCTOR_MAX_AUTOTUNE",
    ]

    # Benchmark the model
    # Feeding in a tensor, and no dataloader, will cause the benchmark_model function to generate a
    # dataloader that provides random tensors of the same shape as `data`, which is used to
    # benchmark the model.
    # NOTE: one needs to squeeze the data tensor to remove the batch dimension
    logging.info("Benchmarking model using random data")
    results: Dict[str, Dict[str, Any]] = benchmark_model(
        pipe,
        config,
        conversions,
        data=None,
        data_loader=data_loader,
    )

    # Display the results
    display_all_results(
        results, display_function=print, include_traceback_for_errors=False
    )


if __name__ == "__main__":
    main()
