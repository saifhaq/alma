# Examples

`alma` will have to be installed for these examples to work.
[See installation steps](../README.md#installation)

## Overview
The examples on how to use `alma` are located here in the `examples` directory. The examples are:

- [`mnist`](./mnist/README.md#mnist-example): This is our most detailed example, showcasing 
different ways to use `alma`. Overall, it has a full example of:
    - Shell scripts for downloading MNIST data.
    - Training a small model on the MNIST dataset.
    - Benchmarking the model using speed for different conversion options. This is where most of 
    the tehcnical documentation of `alma` lives.
    - There are also details on how to quantize the model, for both Eager mode and FX Graph mode, 
    using PTQ and QAT.
- [`linear`](./linear/README.md#simple-linear-example): A barebones example of a linear layer and 
    ReLU activation, where we demonstrate
    just the benchmarking without any training. We only showcase the no-dataloader example in a script.
    Hoewever, we do have a [Jupyter notebook](./linear/notebook.ipynb) that walks through a lot of ways on how to use `alma` and 
    tailor it to one's use case.
