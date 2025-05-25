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
from alma.utils.multiprocessing import lazyload

from data import CircularSampler, PromptDataset

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"

def main() -> None:
    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="INFO")

    args, _ = parse_benchmark_args()

    # Create dataset and circular dataloader. We add a random string input to reduce cache hits on
    # the prompts.
    dataset = PromptDataset(include_random_prefix=True)
    sampler = CircularSampler(dataset, total_samples=2)
    data_loader = DataLoader(dataset, batch_size=2, sampler=sampler)

    # Set the device one wants to benchmark on
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Select HuggingFace model
    model_name = "HuggingFaceTB/SmolLM-135M"
    # model_name = "Qwen/Qwen1.5-0.5B"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Initialise HF transformers pipeline object, which will be our high-level model wrapper
    # we can pass to alma
    # REQUIRES HUGGINGFACE_TOKEN TO BE AN ENVIRONMENTAL VARIABLE AND TO HAVE ACCEPTED THE META LICENSE
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            """`HF_TOKEN` should be provided as an environmental variable, and one 
should accept the Meta terms and conditions to download the LLama 3 8B Instruct model from HuggingFace"""
        )

    # NOTE: we wrap both the model and the pipeline in a lazyload, see /examples/mnist/README.md,
    # section: `Effect on model memory` for a dicsussion on why this is more memory efficient.
    def create_huggingface_pipeline(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float32,
            device_map="auto",
        )

        return TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            max_length=None,  # See issue: https://github.com/huggingface/transformers/issues/19358
            max_new_tokens=5,  # Maximum tokens we allow the LLM to generate
            min_new_tokens=4,  # Minimum tokens we allow the LLM to generate
            num_return_sequences=1,  # How many sequences we want the model to return per prompt
        )

    import psutil
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    baseline_memory = get_memory_usage()
    print(f"Baseline memory: {baseline_memory:.1f} MB\n")

# Method 1: Check the is_loaded() method
    print("=== Method 1: Using is_loaded() method ===")

    pipe = lazyload(lambda: create_huggingface_pipeline(model_name))

    print(f"Model created, is_loaded(): {pipe.is_loaded()}")
    print(f"Memory after creating lazy loader: {get_memory_usage():.1f} MB")
    # pipeline = pipeline.load()
    # print(f"Memory after loading: {get_memory_usage():.1f} MB")

    # Configuration for the benchmarking
    config = BenchmarkConfig(
        n_samples=2,
        batch_size=2,
        device=device,
        multiprocessing=False,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        fail_on_error=True,  # If False, we fail gracefully and keep testing other methods
        non_blocking=False,  # If True, we don't block the main thread when transferring data from host to device
    )

    # Hard-code a list of options. These can be provided as a list of strings, or a list of ConversionOption objects
    conversions = [
        "EAGER",
        # "JIT_TRACE",
        "COMPILE_INDUCTOR_DEFAULT",
        # "COMPILE_OPENXLA",
        # "COMPILE_INDUCTOR_MAX_AUTOTUNE",
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
