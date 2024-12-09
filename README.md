# alma
<p align="center">
  A Python library for benchmarking PyTorch model speed for different conversion options 🚀
</p>
<h2 align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license" style="height: 20px;">
  </a>
  <a href="https://discord.gg/RASFKzqgfZ">
    <img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;">
  </a>
</h2>

The motivation of `alma` is to make it easy for people to benchmark their models for different conversion options,
e.g. eager, tracing, scripting, torch.compile, torch.export, ONNX, Tensort, etc. The library is
designed to be simple to use, with benchmarking provided via a single API call, and to be easily
extensible for adding new conversion options.

Beyond just benchmarking, `alma` is designed to be a one-stop-shop for all model conversion options,
so that one can learn about the different conversion options, how to implement them, and how they
affect model speed and performance.


## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Docker](#docker)
- [Basic Usage](#basic-usage)
- [Examples](#examples)
- [Conversion Options](#conversion-options)
- [Future Work](#future-work)
- [How to Contribute](#how-to-contribute)


## Getting Started

### Installation
`alma` is available as a Python package.

One can install the package from python package index by running 
```bash
pip install alma-torch
```

Alternatively, it can be installed from the root of this repository (save level as this README) by 
running:
```bash
pip install -e .
```

### Docker
We recommend that you build the provided Dockerfile to ensure an easy installation of all of the 
system dependencies and the alma pip packages. 

1. **Build the Docker Image**  
   ```bash
   bash scripts/build_docker.sh
   ```

2. **Run the Docker Container**  
   Create and start a container named `alma`:  
   ```bash
   bash scripts/run_docker.sh
   ```

3. **Access the Running Container**  
   Enter the container's shell:  
   ```bash
   docker exec -it alma bash
   ```

4. **Mount Your Repository**  
   By default, the `run_docker.sh` script mounts your `/home` directory to `/home` inside the container.  
   If your `alma` repository is in a different location, update the bind mount, for example:  
   ```bash
   -v /Users/myuser/alma:/home/alma
   ```


## Basic usage
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


## Examples:

For extensive examples on how to use `alma`, as well as simple clean examples on how train a model and
quantize it, see the [`MNIST example`](./examples/mnist/README.md#overview) directory. These more advanced use cases
include:
- Feeding in a single tensor rather than a dataloader, and having the data tensor implicitly 
initialise an internal data loader inside of `benchmark_model`.
- Using argparser for easy control and experimentation, including selecting conversion methods with
numerical indices.
- Dealing with error handling. If any conversion method fails, `alma` will fail gracefully for that method
and one can access tht error message and traceback from the returned object.
- Debugging and logging. A lot of the conversion methods have very verbose logging. We have opted to
mostly silence those logs. However, if one wants access to those logs, one should use the `setup_logging`
function and set the debugging level to `DEBUG`.

For a short working example on a simple Linear+ReLU, see the [`linear example`](./examples/linear/README.md#overview).

## Conversion Options

### Naming conventions

The naming convention for conversion options is to use short but descriptive names, e.g. `EAGER`, 
`EXPORT+EAGER`, `EXPORT+TENSORRT`, etc. If multiple "techniques" are used in a
single conversion option, then the names are separated by a `+` sign in chronological order of operation. 
Underscores `_` are used within each technique name to seperate the words for readability, 
e.g. `EXPORT+AOT_INDUCTOR`, where `EXPORT` and `AOT_INDUCTOR` are considered seperate steps.
All conversion options are located in the `src/alma/conversions/` directory. Within this directory:


### Code

All conversion options are located in the `src/alma/conversions/` directory. In this directory:

- The `options/` subdirectory contains one Python file per conversion option (or a closely related 
family of options, e.g. torch.compile backends).  
- The main selection logic for these options is found in `select.py`. This is just a glorified 
match-case statement that returns the forward calls of each model conversion option, which is 
returned to the benchmarking loop. It is that simple!

At the risk of some code duplication, we have chosen to keep the conversion options separate, so 
that one can easily add new conversion options without having to modify the existing ones. It also 
makes it easier for the user to see what conversion options are available, and to understand what 
each conversion option does.


### Options Summary
Below is a table summarizing the currently supported conversion options and their identifiers:

  | ID  | Conversion Option                                             |
  |-----|---------------------------------------------------------------|
  | 0   | EXPORT+COMPILE_INDUCTOR                                       |
  | 1   | EXPORT+COMPILE_CUDAGRAPH                                      |
  | 2   | EXPORT+COMPILE_ONNXRT                                         |
  | 3   | EXPORT+COMPILE_OPENXLA                                        |
  | 4   | EXPORT+COMPILE_TVM                                            |
  | 5   | EXPORT+COMPILE_INDUCTOR_EAGER_FALLBACK                        |
  | 6   | EXPORT+AOT_INDUCTOR                                           |
  | 7   | EXPORT+EAGER                                                  |
  | 8   | EXPORT+AI8WI8_STATIC_QUANTIZED                                |
  | 9   | EXPORT+AI8WI8_FLOAT_QUANTIZED                                 |
  | 10  | EXPORT+AI8WI8_STATIC_QUANTIZED+AOT_INDUCTOR                   |
  | 11  | EXPORT+AI8WI8_FLOAT_QUANTIZED+AOT_INDUCTOR                    |
  | 12  | EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION              |
  | 13  | EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION               |
  | 14  | EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR |
  | 15  | EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR  |
  | 16  | COMPILE_INDUCTOR_DEFAULT                                              |
  | 17  | COMPILE_CUDAGRAPH                                             |
  | 18  | COMPILE_ONNXRT                                                |
  | 19  | COMPILE_OPENXLA                                               |
  | 20  | COMPILE_TVM                                                   |
  | 21  | COMPILE_INDUCTOR_EAGER_FALLBACK                                |
  | 22  | EAGER                                                         |
  | 23  | TENSORRT                                                      |
  | 24  | ONNX_CPU                                                      |
  | 25  | ONNX_GPU                                                      |
  | 26  | ONNX+DYNAMO_EXPORT                                            |
  | 27  | NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED                         |
  | 28  | NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC                            |


These conversion options are also all hard-coded in the `alma/conversions/select.py` file, which 
is the source of truth.

## Future work:

- Add more conversion options. This is a work in progress, and we are always looking for more conversion options.
- Multi-device benchmarking. Currently `alma` only supports single-device benchmarking, but ideally a model
  could be split across multiple devices.
- Integrating conversion options beyond PyTorch, e.g. HuggingFace, JAX, llama.cpp, etc.

## How to contribute:

Contributions are welcome! If you have a new conversion option or feature you would like to add, 
so that the whole community can benefit, please open a pull request! We are always looking for new 
conversion options, and we are happy to help you get started with adding a new conversion 
option/feature!

If adding new conversion options, please follow the naming conventions outlined in the [Naming 
Conventions](#naming-conventions) section.


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
