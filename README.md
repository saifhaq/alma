# alma
<p align="center"><img width="900" src="assets/alma-logo-banner.jpg" alt="Alma Logo"></p>
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
  <a href="https://pepy.tech/project/alma-torch">
    <img src="https://pepy.tech/badge/alma-torch?style=flat" alt="Downloads" style="height: 20px;">
  </a>
  <a href="https://pepy.tech/project/alma-torch">
    <img src="https://img.shields.io/pypi/dm/alma-torch" alt="monthly downloads" style="height: 20px;">
  </a>
</h2>

With just one function call, you can get a full report on how fast your PyTorch model runs for inference across over 40 conversion options, such as
JIT tracing, torch.compile, torch.export, torchao, ONNX, OpenVINO, TensorRT, and many more!

This allows one to find the best option for one's model, data, and hardware. See 
[here](#conversion-options) for all supported options.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Docker](#docker)
- [Basic Usage](#basic-usage)
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

Alternatively, it can be installed from the root of this 
[repository](https://github.com/saifhaq/alma) by running:

```bash
pip install -e .
```

### Docker
We recommend that you build the provided Dockerfile to ensure an easy installation of all of the 
system dependencies and the alma pip packages. 

<details>
<summary>Working with the docker image</summary>
<br>

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
</details>


## Basic usage
The core API is `benchmark_model`, which is used to benchmark the speed of a model for different
conversion options. The usage is as follows:

```python
from alma import benchmark_model
from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model
model = ...

# Load the dataloader used in benchmarking
data_loader = ...

# Set the configuration (this can also be passed in as a dict)
config = BenchmarkConfig(
    n_samples=2048,
    batch_size=64,
    device=device,  # The device to run the model on
)

# Choose with conversions to benchmark
conversions = ["EAGER", "TORCH_SCRIPT", "COMPILE_INDUCTOR_MAX_AUTOTUNE", "COMPILE_OPENXLA"]

# Benchmark the model
results = benchmark_model(model, config, conversions, data_loader=data_loader)

# Print all results
display_all_results(results)
```

The results will look like this, depending on one's model, dataloader, and hardware.

```bash
EAGER results:
Device: cuda
Total elapsed time: 0.0206 seconds
Total inference time (model only): 0.0074 seconds
Total samples: 2048 - Batch size: 64
Throughput: 275643.45 samples/second

TORCH_SCRIPT results:
Device: cuda
Total elapsed time: 0.0203 seconds
Total inference time (model only): 0.0043 seconds
Total samples: 2048 - Batch size: 64
Throughput: 477575.34 samples/second

COMPILE_INDUCTOR_MAX_AUTOTUNE results:
Device: cuda
Total elapsed time: 0.0159 seconds
Total inference time (model only): 0.0035 seconds
Total samples: 2048 - Batch size: 64
Throughput: 592801.70 samples/second

COMPILE_OPENXLA results:
Device: xla:0
Total elapsed time: 0.0146 seconds
Total inference time (model only): 0.0033 seconds
Total samples: 2048 - Batch size: 64
Throughput: 611865.07 samples/second
```

See the [examples](./examples) for discussion of design choices and for examples of more advanced usage, e.g. controlling the 
multiprocessing setup, controlling graceful failures, setting default device fallbacks if a conversion
option is incompatible with your specified device, memory efficient usage of `alma`, etc.

## Conversion Options

### Naming conventions

The naming convention for conversion options is as follows:
- Short but descriptive names for each technique, e.g. `EAGER`, `EXPORT`, etc.
- Underscores `_` are used within each technique name to seperate the words for readability, 
e.g. `AOT_INDUCTOR`, `COMPILE_CUDAGRAPHS`, etc.
- If multiple "techniques" are used in a conversion option, then the names are separated by a `+` sign in chronological order of operation. 
    For example, `EXPORT+EAGER`, `EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE`. In both cases, 
    `EXPORT` is the first operation, followed by `EAGER` or `COMPILE_INDUCTOR_MAX_AUTOTUNE`.

### Conversion Options Summary
Below is a table summarizing the currently supported conversion options and their identifiers:

  | ID  | Conversion Option                                 | Device Support | Project |
  |-----|---------------------------------------------------|----------------|---------|
  | 0   |  EAGER                                            | CPU, MPS, GPU  | PyTorch   |
  | 1   |  EXPORT+EAGER                                     | CPU, MPS, GPU  | torch.export |
  | 2   |  ONNX_CPU                                         | CPU            | ONNXRT      |
  | 3   |  ONNX_GPU                                         | GPU            | ONNXRT      |
  | 4   |  ONNX+DYNAMO_EXPORT                               | CPU            | ONNXRT      |
  | 5   |  COMPILE_CUDAGRAPHS                               | GPU (CUDA)     | torch.compile |
  | 6   |  COMPILE_INDUCTOR_DEFAULT                         | CPU, MPS, GPU  | torch.compile |
  | 7   |  COMPILE_INDUCTOR_REDUCE_OVERHEAD                 | CPU, MPS, GPU  | torch.compile |
  | 8   |  COMPILE_INDUCTOR_MAX_AUTOTUNE                    | CPU, MPS, GPU  | torch.compile |
  | 9   |  COMPILE_INDUCTOR_EAGER_FALLBACK                  | CPU, MPS, GPU  | torch.compile |
  | 10  |  COMPILE_ONNXRT                                   | CPU, MPS, GPU  | torch.compile + ONNXRT |
  | 11  |  COMPILE_OPENXLA                                  | XLA_GPU        | torch.compile + OpenXLA |
  | 12  |  COMPILE_TVM                                      | CPU, MPS, GPU  | torch.compile + Apache TVM |
  | 13  |  EXPORT+AI8WI8_FLOAT_QUANTIZED                    | CPU, MPS, GPU  | torch.export |
  | 14  |  EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION  | CPU, MPS, GPU  | torch.export |
  | 15  |  EXPORT+AI8WI8_STATIC_QUANTIZED                   | CPU, MPS, GPU  | torch.export |
  | 16  |  EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION | CPU, MPS, GPU  | torch.export |
  | 17  |  EXPORT+AOT_INDUCTOR                              | CPU, MPS, GPU  | torch.export + aot_inductor |
  | 18  |  EXPORT+COMPILE_CUDAGRAPHS                        | GPU (CUDA)     | torch.export + torch.compile |
  | 19  |  EXPORT+COMPILE_INDUCTOR_DEFAULT                  | CPU, MPS, GPU  | torch.export + torch.compile |
  | 20  |  EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD          | CPU, MPS, GPU  | torch.export + torch.compile |
  | 21  |  EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE             | CPU, MPS, GPU  | torch.export + torch.compile |
  | 22  |  EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK   | CPU, MPS, GPU  | torch.export + torch.compile |
  | 23  |  EXPORT+COMPILE_ONNXRT                            | CPU, MPS, GPU  | torch.export + torch.compile + ONNXRT |
  | 24  |  EXPORT+COMPILE_OPENXLA                           | XLA_GPU        | torch.export + torch.compile + OpenXLA |
  | 25  |  EXPORT+COMPILE_TVM                               | CPU, MPS, GPU  | torch.export + torch.compile + Apache TVM |
  | 26  |  NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED           | CPU            | CPU (PyTorch) |
  | 27  |  NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC              | CPU, GPU       | CPU (PyTorch) |
  | 28  |  COMPILE_TENSORRT                                 | GPU (CUDA)     | torch.compile + NVIDIA TensorRT |
  | 29  |  EXPORT+COMPILE_TENSORRT                          | GPU (CUDA)     | torch.export + torch.compile + NVIDIA TensorRT |
  | 30  |  COMPILE_OPENVINO                                 | CPU (Intel)    | torch.compile + OpenVINO  |
  | 31  |  JIT_TRACE                                        | CPU, MPS, GPU  | PyTorch   |
  | 32  |  TORCH_SCRIPT                                     | CPU, MPS, GPU  | PyTorch   |
  | 33  |  OPTIMUM_QUANTO_AI8WI8                            | CPU, MPS, GPU  | optimum quanto |
  | 34  |  OPTIMUM_QUANTO_AI8WI4                            | CPU, MPS, GPU (not all GPUs supported) | optimum quanto |
  | 35  |  OPTIMUM_QUANTO_AI8WI2                            | CPU, MPS, GPU (not all GPUs supported) | optimum quanto |
  | 36  |  OPTIMUM_QUANTO_WI8                               | CPU, MPS, GPU  | optimum quanto |
  | 37  |  OPTIMUM_QUANTO_WI4                               | CPU, MPS, GPU (not all GPUs supported) | optimum quanto |
  | 38  |  OPTIMUM_QUANTO_WI2                               | CPU, MPS, GPU (not all GPUs supported) | optimum quanto |
  | 39  |  OPTIMUM_QUANTO_Wf8E4M3N                          | CPU, MPS, GPU  | optimum quanto |
  | 40  |  OPTIMUM_QUANTO_Wf8E4M3NUZ                        | CPU, MPS, GPU  | optimum quanto |
  | 41  |  OPTIMUM_QUANTO_Wf8E5M2                           | CPU, MPS, GPU  | optimum quanto |
  | 42  |  OPTIMUM_QUANTO_Wf8E5M2+COMPILE_CUDAGRAPHS        | GPU (CUDA)     | optimum quanto + torch.compile |
  | 43  |  FP16+EAGER                                       | CPU, MPS, GPU  | PyTorch   |
  | 44  |  BF16+EAGER                                       | CPU, MPS, GPU (not all GPUs natively supported)  | PyTorch   |
  | 45  |  COMPILE_INDUCTOR_MAX_AUTOTUNE+<br>TORCHAO_AUTOQUANT_DEFAULT    | GPU  | torch.compile + torchao |
  | 46  |  COMPILE_INDUCTOR_MAX_AUTOTUNE+<br>TORCHAO_AUTOQUANT_NONDEFAULT | GPU  | torch.compile + torchao |
  | 47  |  COMPILE_CUDAGRAPHS+<br>TORCHAO_AUTOQUANT_DEFAULT               | GPU (CUDA) | torch.compile + torchao |
  | 48  |  COMPILE_INDUCTOR_MAX_AUTOTUNE+<br>TORCHAO_QUANT_I4_WEIGHT_ONLY | GPU (requires bf16 support)  | torch.compile + torchao |
  | 49  |  TORCHAO_QUANT_I4_WEIGHT_ONLY                               | GPU (requires bf16 support) | torchao |



These conversion options are also all hard-coded in the [conversion options](src/alma/conversions/conversion_options.py)
file, which is the source of truth.


## Testing:

We use pytest for testing. Simply run:
```bash
pytest
```

We currently don't have comprehensive tests, but we are working on adding more tests to ensure that
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
