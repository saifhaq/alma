# alma

A Python library for benchmarking PyTorch model speed and performance for different conversion
options.

The motivation is to make it easy for people to benchmark their models for different conversion options,
e.g. eager, tracing, scripting, torch.compile, torch.export, ONNX, Tensort, etc. The library is
designed to be simple to use, with benchmarking provided via a single API call, and to be easily
extensible for adding new conversion options.

Beyond just benchmarking, `alma` is designed to be a one-stop-shop for all model conversion options,
so that one can learn about the different conversion options, how to implement them, and how they
affect model speed and performance.

## Usage

The core API is the `benchmark_model` API, which is used to benchmark the speed of a model for different
conversion options. The usage is as follows:

```python
from alma import benchmark_model
from alma.benchmark.log import display_all_results
from typing import Any, Dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model
model = ...

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
results: Dict[str, Dict[str, Any]] = benchmark_model(
    model, config, conversions, data_loader=data_loader
)

# Print all results
display_all_results(results)
```

The results will look like this, depending on one's model, dataloader, and hardware.

```bash
EAGER results:
Device: cuda:0
Total elapsed time: 0.4148 seconds
Total inference time (model only): 0.0436 seconds
Total samples: 5000
Throughput: 12054.50 samples/second

EXPORT+EAGER results:
Device: cuda:0
Total elapsed time: 0.3906 seconds
Total inference time (model only): 0.0394 seconds
Total samples: 5000
Throughput: 12800.82 samples/second
```

### Error handling

In case where the benchmarking of a given conversion fails, it will return a dict for that conversion
which contains the error message as well as the full traceback. This is useful for debugging and
understanding why a given conversion failed, e.g. because of hardware incompatabilities, missing
dependencies, etc.

For example, if the `FAKE_QUANTIZED` conversion fails because it's not currently supported for Apple
silicon, the results may look like this:

```bash
CONVERT_QUANTIZED results:
Device: cpu
Total elapsed time: 0.8287 seconds
Total inference time (model only): 0.4822 seconds
Total samples: 5000 - Batch size: 50
Throughput: 6033.89 samples/second


FAKE_QUANTIZED results:
Benchmarking failed, error: The operator 'aten::_fake_quantize_learnable_per_tensor_affine' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
```

Printing the traceback for the failed `FAKE_QUANTIZED`, stored in the results dict, will give us
the full traceback, which can be useful, e.g. `results["FAKE_QUANTIZED"]["traceback"]` gives us:

```bash
Benchmarking failed, error: The operator 'aten::_fake_quantize_learnable_per_tensor_affine' is not
currently implemented for the MPS device. If you want this op to be added in priority during
the prototype phase of this feature, please comment on
https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the
environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
WARNING: this will be slower than running natively on MPS.
```

### Logging and CI integration

A lot of the imported modules have verbose logging, and so `alma` provides some logging functions
to help manage the logging level. The `setup_logging` function is provided for convenience, but one
can use whatever logging one wishes, or none.
We also provide a `silence_logging` function to silence the logging of all imported modules.

Furthermore, we provide a `display_all_results` function to print the results in a nice format, and a
`save_dict_to_json` function to save the results to a JSON file for easy CI integration.

### Argparsing

For convenience, we provide a `parse_benchmark_args` function that parses the command line arguments
for the user. This is provided for convenience, but one can also just pass in the arguments directly to the
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
--n-samples 5000
```

This will run the EXPORT+EAGER conversion option and the EAGER conversion option, benchmarking the
model speed for each conversion option. The batch size of the data loader is controlled via the
`batch_size` argument. The number of samples to run the benchmark on is controlled via the `n_samples`
argument.

All of the command line arguments are optional, subject to one using the `parse_benchmark_args` API.
The defaults are set in `alma/arguments/benchmark_args.py`. These include standard arguments for
convenience, e.g. `--model-path` for model weight loading, and `--data-dir` for data loading.
One can of course feed in one's variables straight into the `benchmark_model` API.

To see all of the arguments, run the following command:

```bash
cd examples/mnist
python benchmark_with_dataloader.py --help
```

## Examples:

For extensive examples on how to use `alma`, as well as simple clean examples on how train a model and
quantize it, see the `examples` directory.

## Conversion options:

The currently supported conversion options are:

```bash
XXX
YYY
```

These conversion options are also all hard-coded in the `alma/conversions/select.py` file, which
is the source of truth.

#### Naming conventions

The convention for naming convention options in the match-case statement is to use the names of the conversion
steps, e.g. `EAGER`, `EXPORT+EAGER`, `EXPORT+TENSORRT`, etc. If multiple "techniques" are used in a
single conversion option, then the names are separated by a `+` sign. Underscores `_` are used within each
technique name to seperate the words for readability, e.g. `EXPORT+AOT_INDUCTOR`.

## Examples:

For extensive examples on how to use `alma`, as well as simple clean examples on how train a model and
quantize it, see the `examples` directory.

## Future work:

- Add more conversion options. This is a work in progress, and we are always looking for more conversion options.
- Multi-device benchmarking. Currently `alma` only supports single-device benchmarking, but ideally a model
  could be split across multiple devices.
- A self-contained web application for benchmarking. This would allow users to spin up a web server
  and benchmark their models in a more user-friendly way, with `alma` contained in a Docker image.
  This would also allow for easy deployment on cloud services, and the user wouldn't have to worry about
  installing dependencies.

## How to contribute:

Contributions are welcome! If you have a new conversion option you would like to add, or a new feature,
please open a pull request. We are always looking for new conversion options, and we are happy to help
you get started with adding a new conversion option.

### Conversion Options

All conversion options are set in `src/alma/conversions/`. In that directory, one can find the conversion
options inside `options/`, where each file contains a single conversion option. At the risk of some code
duplication, we have chosen to keep the conversion options separate, so that one can easily add new conversion
options without having to modify the existing ones. It also makes it easier for the user to see what conversion
options are available, and to understand what each conversion option does.

The conversion options are then selected for benchmarking in the `src/alma/conversions/select.py` file.
This is just a glorified match-case statement that returns the forward calls of each model conversion option,
which is returned to the benchmarking loop. It is that simple!

If adding any conversion options, follow the naming convention in [Naming Conventions](#naming-conventions).

### Dependencies

Some conversion options may require package dependencies, which should be added to the Docker image.

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
