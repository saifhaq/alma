# alma
A library for benchmarking PyTorch models for different conversion options.

## Benchmarking

The `benchmark_model` API is used to benchmark the speed of a model for different conversion options.
The API is used as follows:

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args

# Parse the arguments, e.g. the model path, device, and conversion options
# This is provided for convenience, but one can also just pass in the arguments directly to the
# `benchmark_model` API.
args, device = parse_benchmark_args()

# Load the model
model = ...

# Load the data
data_loader = ...

# Benchmark the model
benchmark_model(
    model, device, args, args.conversions, data_loader=data_loader
)
```

One can then run the sxript from the command line with the following command:
```bash
python YOUR_BENCHMARK_SCRIPT.py --conversions EAGER,EXPORT+EAGER --batch-size 10
--n-samples 5000 
```

This will run the EXPORT+EAGER conversion option and the EAGER conversion option, benchmarking the
model speed for each conversion option. The batch size of the data loader is controlled via the
`batch_size` argument. The number of samples to run the benchmark on is controlled via the `n_samples`
argument. 

The results will look like this, depending on one's model, dataloader and hardware.

```bash

EAGER results:
Total elapsed time: 0.4148 seconds
Total inference time (model only): 0.0436 seconds
Total samples: 5000
Throughput: 12054.50 samples/second

EXPORT+EAGER results:
Total elapsed time: 0.3906 seconds
Total inference time (model only): 0.0394 seconds
Total samples: 5000
Throughput: 12800.82 samples/second

```

All of the command line arguments are optional, subject to one using the `parse_benchmark_args` API.
The defaults are set in `alma/arguments/benchmark_args.py`. These include standard arguments for 
convenience, e.g. `--model-path` for model weight loading, and `--data-dir` for data loading.
One can of course feed in one's variables straight into the `benchmark_model` API.

The see all of the arguments, run the following command:
```bash
cd examples/mnist
python benchmark_with_dataloader.py --help
```

This will also show all of the model conversion options, which include:
```bash
XXX
YYY
```


## Examples:
For extensice examples on how to use `alma`, as well as simple clean examples on how train a model and
quantize it, see the `examples` directory.

## How to contribute:

### Conversion Options
All conversion options are set in `src/alma/conversions/`. In that directory, one can find the conversion
options inside `options/`, where each file contains a single conversion option. At the risk of some code
duplication, we have chosen to keep the conversion options separate, so that one can easily add new conversion
options without having to modify the existing ones. It also makes it easier for the user to see what conversion
options are available, and to understand what each conversion option does.

The conversion options are then selected for benchmarking in the `src/alma/conversions/select.py` file. 
This is just a glorified match-case statement that returns the forward calls of each model conversion option,
which is returned to the benchmarking loop. It is that simple!

#### Naming conventions
The convention for naming convention options in the match-case statement is to use the names of the conversion
steps, e.g. `EAGER`, `EXPORT+EAGER`, `EXPORT+TENSORRT`, etc. If multiple "techniques" are used in a
single conversion option, then the names are separated by a `+` sign. Underscores `_` are used within each
technique name to seperate the words for readability, e.g. `EXPORT+AOT_INDUCTOR`.

#### Dependencies
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

