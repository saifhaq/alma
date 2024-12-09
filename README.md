# alma

A Python library for benchmarking PyTorch model speed for different conversion options.

The motivation is to make it easy for people to benchmark their models for different conversion options,
e.g. eager, tracing, scripting, torch.compile, torch.export, ONNX, Tensort, etc. The library is
designed to be simple to use, with benchmarking provided via a single API call, and to be easily
extensible for adding new conversion options.

Beyond just benchmarking, `alma` is designed to be a one-stop-shop for all model conversion options,
so that one can learn about the different conversion options, how to implement them, and how they
affect model speed and performance.

## Installation
`alma` is available as a Python package.

One can install the package from python package index by running 
```bash
pip install alma
```

Alternatively, it can be installed from the root of this repository (save level as this README) by 
running:
```bash
pip install -e .
```


## Usage

The core API is `benchmark_model`, which is used to benchmark the speed of a model for different
conversion options. The usage is as follows:

```python
from alma import benchmark_model
from alma.benchmark.log import display_all_results

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model
model = ...
model = model.to(device)

# Load the dataloader used in benchmarking
data_loader = ...

# Set the configuration
config = {
    "batch_size": 128,
    "n_samples": 4096,
}

# Choose with conversions to benchmark:
conversions = ["EAGER", "EXPORT+EAGER"]

# Benchmark the model
results = benchmark_model(model, config, conversions, data_loader=data_loader)

# Print all results
display_all_results(results)
```

The results will look like this, depending on one's model, dataloader, and hardware.

```bash
EAGER results:
device: cuda:0
Total elapsed time: 0.4148 seconds
Total inference time (model only): 0.0436 seconds
Total samples: 5000
Throughput: 12054.50 samples/second

EXPORT+EAGER results:
device: cuda:0
Total elapsed time: 0.3906 seconds
Total inference time (model only): 0.0394 seconds
Total samples: 5000
Throughput: 12800.82 samples/second
```

### Feeding in a single `data` tensor instead of a dataloader
We also provide the option to feed in a single `data` tensor instead of a dataloader. This is useful
for cases where one does not want to go through the trouble of setting up a dataloader, and is happy
to just benchmark the model on random data of a given shape.
If no `data_loader` argument is provided and a `data` tensor is fed in, `benchmark_model`
will automatically generate a dataloader of random tensors, of the same shape as `data`. 
`data` should be fed in WITHOUT a batch dimension.
The batch size of the dataloader will be equal to the `batch_size` in the `config` dict. 

```python
from alma import benchmark_model
from alma.benchmark.log import display_all_results

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model
model = ...
model = model.to(device)

# Initialise a random tensor to benchmark the model on. It must have batch size of 1 (and squeezed)
# or no batch dimension.
data = torch.randn(1, 3, 224, 224).to(device)

# Set the configuration
config = {
    "batch_size": 128,
    "n_samples": 4096,
}

# Choose with conversions to benchmark:
conversions = ["EAGER", "EXPORT+EAGER"]

# Benchmark the model
results = benchmark_model(model, config, conversions, data=data.squeeze())

# Print all results
display_all_results(results)
```


### Error handling

In case where the benchmarking of a given conversion fails, it will return a dict for that conversion
which contains the error message as well as the full traceback. This is useful for debugging and
understanding why a given conversion failed, e.g. because of hardware incompatabilities, missing
dependencies, etc. 

For example, if the `FAKE_QUANTIZED` conversion fails because it's not currently supported for Apple
silicon, the default results may look like this:

```bash
CONVERT_QUANTIZED results:
Device: cpu
Total elapsed time: 0.8287 seconds
Total inference time (model only): 0.4822 seconds
Total samples: 5000 - Batch size: 50
Throughput: 6033.89 samples/second


FAKE_QUANTIZED results:
Benchmarking failed, error: The operator 'aten::_fake_quantize_learnable_per_tensor_affine' is not 
currently implemented for the MPS device. If you want this op to be added in priority during the 
prototype phase of this feature, please comment on 
https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment 
variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this 
will be slower than running natively on MPS.
```

The traceback of the error is also stored in the results dict, and can be accessed via 
`results[CONVERSION_NAME]["traceback"]`.

In this example, `print(results["FAKE_QUANTIZED"]["traceback"])` gives us:

```bash
Traceback (most recent call last):
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark_model.py", line 88, in benchmark_model
    result: Dict[str, float] = benchmark(
                               ^^^^^^^^^^
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark/benchmark.py", line 63, in benchmark
    warmup(forward_call, data_loader, device)
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark/warmup.py", line 26, in warmup
    _ = forward_call(data)
        ^^^^^^^^^^^^^^^^^^
  File "<eval_with_key>.173", line 5, in forward
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/quantization-tuts/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/quantization-tuts/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/quantization-tuts/lib/python3.11/site-packages/torch/ao/quantization/_learnable_fake_quantize.py", line 160, in forward
    X = torch._fake_quantize_learnable_per_tensor_affine(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NotImplementedError: The operator 'aten::_fake_quantize_learnable_per_tensor_affine' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
```

By default `display_all_results` only logs the error from the conversion, but one can also 
include the traceback in `display_all_results` via the `include_traceback_for_errors` argument
E.g. `display_all_results(results, include_traceback_for_errors=True)`.

Incidentally, this Apple-silicon-quantization issue can be solved by setting the environmental variable:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Logging and CI integration

A lot of the imported modules have verbose logging, and so `alma` provides some logging functions
to help manage the logging level. The `setup_logging` function is provided for convenience, but one
can use whatever logging one wishes, or none.
We also provide a `silence_logging` function to silence the logging of all imported modules.

Furthermore, as we have highlighted prior, we provide a `display_all_results` function to print 
the results in a nice format.There is also a `save_dict_to_json` function to save the results to a 
JSON file for easy CI integration.

If one is debugging, it is highly recommended that one use the `setup_logging` function and set one's
level to DEBUG. This will, among other things, log any torch.compile warnings and errors thrown by
torch.inductor that can point to isses in triton kernels, and print the model graphs where 
appropriate.

### Argparsing

For convenience, we provide a `parse_benchmark_args` function that parses the command line arguments
for the user. One can of course also just pass in non-CLI arguments directly to the
`benchmark_model` API. The `parse_benchmark_args` function is used as follows:

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args
from alma.utils.setup_logging import setup_logging
from typing import Dict

# A `setup_logging` function is provided for convenience, but one can use whatever logging one
# wishes, or none. DEBUG level will also log the model graphs.
setup_logging(level="INFO")

# Parse the arguments, e.g. the model path, device, and conversion options
# This is provided for convenience, but one can also just pass in the arguments directly to the
# `benchmark_model` API.
args, device = parse_benchmark_args()

# Load the model
model = ...
model = model.to(device)

# Load the data
data_loader = ...

# Set the configuration
config = {
    "batch_size": args.batch_size,
    "n_samples": args.n_samples,
}

# Benchmark the model
results: Dict[str, Dict[str, float]] = benchmark_model(
    model, config, args.conversions, data_loader=data_loader
)
```

One can then run the script from the command line with the following command:

```bash
python YOUR_BENCHMARK_SCRIPT.py --conversions EAGER,EXPORT+EAGER --batch-size 10
--n-samples 5000 --ipdb
```

This will run the EXPORT+EAGER conversion option and the EAGER conversion option, benchmarking the
model speed for each conversion option. 
The batch size of the data loader is controlled via then`batch_size` argument. 
The number of samples to run the benchmark on is controlled via the `n_samples` argument. 
The `--ipdb` creates a magic breakpoint that throws one into an ipdb debugging session if and wherever
an Exception occurs in one's code, making debugging much easier (see this 
[blog post](https://medium.com/@oscar-savolainen/my-favourite-python-snippets-794d5653af38)).

All of the command line arguments are optional, subject to one using the `parse_benchmark_args` API.
The defaults are set in `alma/arguments/benchmark_args.py`. These include standard arguments for
convenience, e.g. `--model-path` for model weight loading, and `--data-dir` for data loading.

To see all of the arguments, run the following command:

```bash
cd examples/mnist
python benchmark_with_dataloader.py --help
```

## Examples:

For extensive examples on how to use `alma`, as well as simple clean examples on how train a model and
quantize it, see the [`examples`](./examples/README.md#overview) directory.


## Conversion options:

### Naming conventions

The naming convention for conversion options is to use short but descriptive names, e.g. `EAGER`, 
`EXPORT+EAGER`, `EXPORT+TENSORRT`, etc. If multiple "techniques" are used in a
single conversion option, then the names are separated by a `+` sign in chronological order of operation. 
Underscores `_` are used within each technique name to seperate the words for readability, 
e.g. `EXPORT+AOT_INDUCTOR`, where `EXPORT` and `AOT_INDUCTOR` are considered seperate steps.


### Current options:

The currently supported conversion options are:

```bash
EXPORT+COMPILE_INDUCTOR_DEFAULT
EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD
EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE
EXPORT+COMPILE_CUDAGRAPH
EXPORT+COMPILE_ONNXRT
EXPORT+COMPILE_OPENXLA
EXPORT+COMPILE_TVM
EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK
EXPORT+AOT_INDUCTOR
EXPORT+EAGER
EXPORT+AI8WI8_STATIC_QUANTIZED
EXPORT+AI8WI8_FLOAT_QUANTIZED
EXPORT+AI8WI8_STATIC_QUANTIZED+AOT_INDUCTOR
EXPORT+AI8WI8_FLOAT_QUANTIZED+AOT_INDUCTOR
EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION
EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION
EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR
EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR
COMPILE_INDUCTOR_DEFAULT
COMPILE_INDUCTOR_REDUCE_OVERHEAD
COMPILE_INDUCTOR_MAX_AUTOTUNE
COMPILE_CUDAGRAPH
COMPILE_ONNXRT
COMPILE_OPENXLA
COMPILE_TVM
COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK
EAGER
TENSORRT
ONNX_CPU
ONNX_GPU
ONNX+DYNAMO_EXPORT
NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED
NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC
```

These conversion options are also all hard-coded in the `alma/conversions/select.py` file, which
is the source of truth.

## Future work:

- Add more conversion options. This is a work in progress, and we are always looking for more conversion options.
- Multi-device benchmarking. Currently `alma` only supports single-device benchmarking, but ideally a model
  could be split across multiple devices.
- Integrating conversion options beyond PyTorch, e.g. HuggingFace, JAX, llama.cpp, etc.

## How to contribute:

Contributions are welcome! If you have a new conversion option or feature you would like to add, so that the whole community can benefit,
please open a pull request! We are always looking for new conversion options, and we are happy to help
you get started with adding a new conversion option/feature!

### Conversion Options

All conversion options are set in `src/alma/conversions/`. In that directory, one can find the conversion
option code inside `options/`, where each file contains a conversion option (or sometimes closely related 
family of options). At the risk of some code duplication, we have chosen to keep the conversion options 
separate, so that one can easily add new conversion options without having to modify the existing ones. 
It also makes it easier for the user to see what conversion options are available, and to understand what 
each conversion option does.

The conversion options are then selected for benchmarking in the `src/alma/conversions/select.py` file.
This is just a glorified match-case statement that returns the forward calls of each model conversion option,
which is returned to the benchmarking loop. It is that simple!

If adding any conversion options, follow the naming convention in [Naming Conventions](#naming-conventions).

### Dependencies

Some conversion options may require package dependencies, which should be added to the Docker image and/or
`requirements.txt` file.

### Code Standards

- **Black**: Ensures consistency following a strict subset of PEP 8.
- **isort**: Organizes Python imports systematically.

#### Automatic Formatting Before Commit

1. From the repo root, run:

```bash
pre-commit install
```

#### Manually Running Hooks

If you want to manually run all hooks on all files, you can do:

```bash
git stage .
pre-commit run --all-files
```
