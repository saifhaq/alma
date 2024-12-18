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
- [Advanced Features and Design Decisions](#advanced-features-and-design-decisions)
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

# Load the dataloader used in benchmarking
data_loader = ...

# Set the configuration
config = {
    "batch_size": 64,
    "n_samples": 2048,
    "device": device,
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
Device: cuda
Total elapsed time: 0.0211 seconds
Total inference time (model only): 0.0073 seconds
Total samples: 2048 - Batch size: 64
Throughput: 282395.70 samples/second


EXPORT+EAGER results:
Device: cuda
Total elapsed time: 0.0209 seconds
Total inference time (model only): 0.0067 seconds
Total samples: 2048 - Batch size: 64
Throughput: 305974.83 samples/second
```


## Examples:

For extensive examples on how to use `alma`, as well as simple examples on how train a model and
quantize it, see the [`MNIST example`](./examples/mnist/README.md#overview) directory. This contains
code examples for all of the different `alma` features, and is where one can find examples on every
feature. 

For a short working example on a simple Linear+ReLU, see the [`linear example`](./examples/linear/README.md#overview).

## Advanced Features and Design Decisions

`alma` is designed to be simple to use, with a single API call to benchmark a model for different
conversion options. Below are some features we have produced and some design decisions we have 
made, which are all configurable by the user. For examples on how to use these features, see the 
[`MNIST example`](./examples/mnist/README.md#overview).

<details>
<summary>Implicitly initialise a data loader inside of `benchmark_model`</summary>
<br>
Rather than initializing and feeding in a data loader like in the above example, one can also 
just pass in a `data` tensor (with no batch dimension), and `benchmark_model` will automatically
create a dataloader that produces random tensors of the same shape as the input tensor, with the batch size
controlled via the `config` dictionary. This can be convenient if one does not want to create a data loader.

See <a href="./examples/mnist/README.md#implicitly-initialise-a-data-loader-inside-of-benchmark_model">here</a> for details.
</details>

<details>
<summary>Pre-defined argparser for easy control and experimentation
</summary>
<br>
We provide an argparser that allows one to easily select conversion methods by numerical index or 
string name. It also allows one to set the batch size, number of samples, and device easily, as well
as other commonly used parameters like model weights path.

See <a href="./examples/mnist/README.md#pre-defined-argparser-for-easy-control-and-experimentation">here</a> for details.
</details>

<details>
<summary>Graceful or fast failure</summary>
<br>
By default, `alma` will fail fast if any conversion method fails. This is because we want to know
if a conversion method fails, so that we can fix it. 
However, if one wants to continue benchmarking other options even if a conversion method fails, 
one can set `fail_on_error` to False in the config dictionary.
`alma` will then fail gracefully for that method. One can then access the associated error messages 
and full tracebacks for the failed methods from the returned object.

See <a href="./examples/mnist/README.md#graceful-or-fast-failure">here</a> for details.
</details>

<details>
<summary>Isolated environments for each conversion method via multi-processing</summary>
<br>
By default, `alma` will run each conversion method in a separate process (one at a time), so that one can benchmark
each conversion method in isolation. This ensures that each conversion method is benchmarked
in a fair and isolated environment, and is relevant because some of the methods (e.g. optimum quanto)
can affect the global torch state and break other methods (e.g. by overwriting tensor defaults in 
the C++ backend).

To disable multiprocessing, set `multiprocessing` to False in the config dictionary.

See <a href="./examples/mnist/README.md#isolated-environments-for-each-conversion-method-via-multi-processing">here</a> for details and discussion.
</details>

<details>
<summary>Logging, debugging, and CI integration</summary>
<br>
A lot of the conversion methods have verbose internal logging. We have opted to
mostly silence those logs. However, if one wants access to those logs, one should use the `setup_logging`
function and set the debugging level to `DEBUG`.

See <a href="./examples/mnist/README.md#logging-and-ci-integration">here</a> for details.
</details>


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

  | ID  | Conversion Option                                 |
  |-----|---------------------------------------------------|
  | 0   |  EAGER                                            |
  | 1   |  EXPORT+EAGER                                     |
  | 2   |  ONNX_CPU                                         |
  | 3   |  ONNX_GPU                                         |
  | 4   |  ONNX+DYNAMO_EXPORT                               |
  | 5   |  COMPILE_CUDAGRAPHS                               |
  | 6   |  COMPILE_INDUCTOR_DEFAULT                         |
  | 7   |  COMPILE_INDUCTOR_REDUCE_OVERHEAD                 |
  | 8   |  COMPILE_INDUCTOR_MAX_AUTOTUNE                    |
  | 9   |  COMPILE_INDUCTOR_EAGER_FALLBACK                  |
  | 10  |  COMPILE_ONNXRT                                   |
  | 11  |  COMPILE_OPENXLA                                  |
  | 12  |  COMPILE_TVM                                      |
  | 13  |  EXPORT+AI8WI8_FLOAT_QUANTIZED                    |
  | 14  |  EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION  |
  | 15  |  EXPORT+AI8WI8_STATIC_QUANTIZED                   |
  | 16  |  EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION |
  | 17  |  EXPORT+AOT_INDUCTOR                              |
  | 18  |  EXPORT+COMPILE_CUDAGRAPHS                        |
  | 19  |  EXPORT+COMPILE_INDUCTOR_DEFAULT                  |
  | 20  |  EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD          |
  | 21  |  EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE             |
  | 22  |  EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK   |
  | 23  |  EXPORT+COMPILE_ONNXRT                            |
  | 24  |  EXPORT+COMPILE_OPENXLA                           |
  | 25  |  EXPORT+COMPILE_TVM                               |
  | 26  |  NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED           |
  | 27  |  NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC              | 
  | 28  |  COMPILE_TENSORRT                                 |
  | 29  |  EXPORT+COMPILE_TENSORRT                          |
  | 30  |  JIT_TRACE                                        |
  | 31  |  TORCH_SCRIPT                                     |
  | 32  |  OPTIMIM_QUANTO_AI8WI8                            |
  | 33  |  OPTIMIM_QUANTO_AI8WI4                            |
  | 34  |  OPTIMIM_QUANTO_AI8WI2                            |
  | 35  |  OPTIMIM_QUANTO_WI8                               |
  | 36  |  OPTIMIM_QUANTO_WI4                               |
  | 37  |  OPTIMIM_QUANTO_WI2                               |
  | 38  |  OPTIMIM_QUANTO_Wf8E4M3N                          |
  | 39  |  OPTIMIM_QUANTO_Wf8E4M3NUZ                        |
  | 40  |  OPTIMIM_QUANTO_Wf8E5M2                           |
  | 41  |  OPTIMIM_QUANTO_Wf8E5M2+COMPILE_CUDAGRAPHS        |


These conversion options are also all hard-coded in the `alma/conversions/select.py` file, which 
is the source of truth.


## Testing

We use pytest for testing. Simply run:
```bash
pytest
```

We currently don't have extensive tests, but we are working on adding more tests to ensure that
the conversion options are working as expected in known environments (e.g. the Docker container).

## Future work:

- Add more conversion options. This is a work in progress, and we are always looking for more conversion options.
- Multi-device benchmarking. Currently `alma` only supports single-device benchmarking, but ideally a model
  could be split across multiple devices.
- Integrating conversion options beyond PyTorch, e.g. HuggingFace, JAX, llama.cpp, etc.

## How to contribute:

Contributions are welcome! If you have a new conversion option, feature, or other you would like to add, 
so that the whole community can benefit, please open a pull request! We are always looking for new 
conversion options, and we are happy to help you get started with adding a new conversion 
option/feature!

See the [CONTRIBUTING.md](./CONTRIBUTING.md) file for more detailed information on how to contribute.


## Citation
```bibtex
@Misc{alma,
  title =        {Alma: PyTorch model speed benchmarking across all conversion types},
  author =       {Oscar Savolainen and Saif Haq},
  howpublished = {\url{https://github.com/saifhaq/alma}},
  year =         {2024}
}
```
