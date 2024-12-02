# Examples

To run the examples, one has to install the `alma` package. This can be installed from the
root of the repository by running `pip install -e .`. Alternatively, one can install the package
from python package index by running `pip install alma`.

The examples are located in the `examples` directory. The examples are:

- `mnist`: A simple example of training a model on the MNIST dataset and benchmarking the model
  speed for different conversion options. There are also many details on how to quantize the model,
  for both Eager mode and FC Graph mode.

## Installation of alma

`cd` to the root of the repository and run:
```bash
pip install -e .
```

Alternatively, one can install the package from python package index by running:
```bash
pip install alma
```

## Running example:

### Getting the data:

To get data for this example, run the following command:

```bash
cd examples/mnist/data
./setup-data.sh
```

To delete the data, run the following command:

```bash
cd examples/mnist/data
./reset.sh
```

## Training model:

To train the model, run the following command:

```bash
cd examples/mnist
python train.py --save-path ./model/mnist.pt
```

See the benchmark script for more details on how to customize the benchmarking process to a given
model and data.

## Benchmark:

### Benchmark conversion options using a default data loader:

The `benchmark_model` API is used to benchmark the speed of a model for different conversion options.
The API is used as follows:

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args

# Parse the arguments, e.g. the model path, device, and conversion options
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

One would run this script (e.g., if one wanted to benchmark the same model as used in training)
with the following command:

```bash
cd examples/mnist
python YOUR_SCRIPT_NAME.py --model-path ./model/mnist.pt --conversions EAGER,EXPORT+EAGER --batch-size 10
--n-samples 5000 --data-dir data/data_for_inference
```

This will run the EXPORT+EAGER conversion option and the EAGER conversion option, benchmarking the
model speed for each conversion option. The batch size of the data loader is controlled via the
`batch_size` argument. The number of samples to run the benchmark on is controlled via the `n_samples`
argument. For convenience, we also provide a `data-dir` argument, so that one can have one's
data loader feed in specific data.

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

An example can be found in `examples/mnist/benchmark.py`.
E.g.

```bash
cd examples/mnist
python benchmark.py --conversion 10,2 --batch-size 100 --n-samples 5000 --data-dir data/data_for_inference --model-path mnist-model.pt
```

To see all of the conversion options, run the following command:

```bash
cd examples/mnist
python YOUR_SCRIPT_NAME.py --help
```

The conversion options can be selected as either an integer or a string. The integer corresponds to the
index of the conversion option in the list of all conversion options. The string corresponds to the name
of the conversion option. These can be mixed and matched.

To feed in multiple conversion options, separate the conversion options with a comma. For example, one
can run:

```bash
cd examples/mnist
python YOUR_SCRIPT_NAME.py --model-path ./model/mnist.pt --conversions 2,EAGER
```

This will run the 2nd conversion option and the EAGER conversion option.

If no conversion options are provided, the script will run all conversion options. For example, one can
run:

```bash
cd examples/mnist
python YOUR_SCRIPT_NAME.py --model-path ./model/mnist.pt
```

### Benchmark conversion options using a data tensor to intialise a data loader:

One does not need to pass in a data loader to do the benchmarking. If one does not pass in a data loader,
one has to pass in a `data` input, where `data` is just a tensor with batch size 1.
The benchmarking will generate a data loader, where the size of the generated tensors is
taken from an inputted `data` tensor, which can be random. Either the `data` tensor or the `data_loader`
has to be provided. The batch size of the generated data loader is controlled via the `batch_size` argument.

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args

# Parse the arguments, e.g. the model path, device, and conversion options
args, device = parse_benchmark_args()

# Load the model
model = ...

# Benchmark the model
benchmark_model(
    model, device, args, args.conversions, data=torch.randn(1, 1, 28, 28)
)
```

One would then run this the same way as before, e.g.

```bash
cd examples/mnist
python YOUR_SCRIPT_NAME.py --model-path ./model/mnist.pt --conversions EAGER,EXPORT+EAGER --batch-size 10
```

An example can be found in `examples/mnist/benchmark_rand_tensor.py` where a random model is used.
E.g.
```bash
cd examples/mnist
python benchmark_rand_tensor.py --conversion 10,2 --batch-size 100 --n-samples 5000
```
