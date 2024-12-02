# Examples

## Installation of alma
To run the examples, one has to install the `alma` package. 

One can install the package from python package index by running 
```bash
pip install alma
```

Alternatively, it can be installed from the root of the repository by running:
```bash
pip install -e .
```

## Overview
The examples on how to use `alma` are located here in the `examples` directory. The examples are:

- `mnist`: A simple example of training a model on the MNIST dataset and benchmarking the model
    speed for different conversion options. There are also many details on how to quantize the model,
    for both Eager mode and FX Graph mode, using PTQ and QAT.
- `random_linear`: A barebones example of a linear layer and ReLU activation, where we demonstrate
    just the benchmarking without any training. 

