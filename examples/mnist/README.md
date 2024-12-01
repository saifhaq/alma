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

# Benchmark:

To benchmark the model speed for different conversion options, run the following command:

```bash
cd examples/mnist
python benchmark.py --model-path ./model/mnist.pt
```

See the benchmark script for more details on how to customize the benchmarking process to a given
model and data.
