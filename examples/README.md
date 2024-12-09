# Examples

`alma` will have to be installed for these examples to work.
[See installation steps](../README.md#installation)

## Overview
The examples on how to use `alma` are located here in the `examples` directory. The examples are:

- [`mnist`](./mnist/README.md#mnist-example): A simple example of training a model on the MNIST dataset and benchmarking the model
    speed for different conversion options. There are also many details on how to quantize the model,
    for both Eager mode and FX Graph mode, using PTQ and QAT.
- [`linear`](./linear/README.md#simple-linear-example): A barebones example of a linear layer and ReLU activation, where we demonstrate
    just the benchmarking without any training. 


### Basic usage
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


