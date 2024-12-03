# MNIST example

This example of `alma` demonstrates how to train a simple model on the MNIST dataset and benchmark
the model speed for different conversions using the `benchmark_model` API.
It also contains a script to download the MNIST dataset, and extensive example code on how to quantize
a model using PTQ and QAT.

## Getting the data:

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

One can also train a quantized (PTQ and QAT) model with the `quantize` argument, e.g.

```bash
python train.py  --quantize
```

This will do fake-quantization on the model.

To quantize the model running on Apple silicon, run:

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py  --quantize
```

## Benchmark with `alma`:

### Benchmark conversion options using a default data loader:

The `benchmark_model` API is used to benchmark the speed of a model for different conversion options.
The API is used as follows (see `benchmark_with_dataloader.py`):

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

# Set the configuration
config = {
    "batch_size": args.batch_size,
    "n_samples": args.n_samples,
}

# Benchmark the model
benchmark_model(
   model, config, args.conversions, data_loader=data_loader
)
```

One would run this script (e.g., if one wanted to benchmark the same model as used in training)
with the following commands (all steps from loading data, to training, to benchmarking):

```bash
python ABOVE_SCRIPT.py --model-path ./model/mnist.pt --conversions EAGER,EXPORT+EAGER --batch-size 10
--n-samples 5000 --data-dir data/data_for_inference
```

This will run the EXPORT+EAGER conversion option and the EAGER conversion option, benchmarking the
model speed for each conversion option. The batch size of the data loader is controlled via the
`batch_size` argument, fed in via the config object. The number of samples to run the benchmark on 
is controlled via the `n_samples` argument. For convenience, we also provide a `data-dir` 
argument, so that one can have one's data loader feed in specific data, and a `model-path` 
argument, so that one can feed in specific model weights.

The results will look like this, depending on one's model, dataloader, hardware, and logging.

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

#### Full working example:
A full working example can be found in `benchmark_with_dataloader.py`.
With the above command, the script will download the MNIST dataset, train a model, and then benchmark
the model for the 2nd and 10th conversion options:
```bash
cd examples/mnist/data
./setup-data.sh
cd examples/mnist
python train.py --save-path ./model/mnist.pt
python benchmark_with_dataloader.py --model-path ./model/mnist.pt --conversion 10,2 --batch-size
100 --n-samples 5000 --data-dir data/data_for_inference
cd examples/mnist/data
./reset.sh
```

This option also has some logging options, so the user can set up logging as desired.
`alma` comes with a default logger that can be imported, or the user can set up their own logger.

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args
from alma.utils.setup_logging import setup_logging
from typing import Dict

# Set up logging (comment out for no logging, or set DEBUG for more logging)
# A `setup_logging` function is provided for convenience, but one can use whatever logging one 
# wishes, or none.
setup_logging(level="INFO")

# Parse the arguments, e.g. the model path, device, and conversion options
# This is provided for convenience, but one can also just pass in the arguments directly to the
# `benchmark_model` API.
args, device = parse_benchmark_args()
    
# Load the model
model = ...

# Load the data
data_loader = ...

config = {
    "batch_size": args.batch_size,
    "n_samples": args.n_samples,
}

# Benchmark the model
results: Dict[str, Dict[str, float]] = benchmark_model(
   model, config, args.conversions, data_loader=data_loader
)
```

### Selecting conversion options:

To see all of the conversion options, run the following command:

```bash
cd examples/mnist
python benchmark_with_dataloader.py --help
```

The conversion options can be selected as either an integer or a string. The integer corresponds to the
index of the conversion option in the list of all conversion options. The string corresponds to the name
of the conversion option. These can be mixed and matched.

To feed in multiple conversion options, separate the conversion options with a comma. For example, one
can run:

```bash
cd examples/mnist
python benchmark_with_dataloader.py --model-path ./model/mnist.pt --conversions 2,EAGER --data-dir 
data/data_for_inference
```

This will run the 2nd conversion option and the EAGER conversion option.

If no conversion options are provided, the script will run all conversion options. For example, one can
run:

```bash
cd examples/mnist
python benchmark_with_dataloader.py --model-path ./model/mnist.pt --data-dir data/data_for_inference
```

### Benchmark conversion options using a data tensor to intialise a data loader:

One does not need to pass in a data loader to do the benchmarking. Howeverm, if one does not pass in a data loader,
one has to pass in a `data` input, where `data` is just a tensor with batch size 1.
The benchmarking will generate a data loader, where the size of the generated tensors is
taken from an inputted `data` tensor, which can be a tensor with random values. Either the `data`
tensor or the `data_loader` has to be provided. The batch size of the generated data loader is
controlled via the `batch_size` argument.

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args
from alma.utils.setup_logging import setup_logging
from typing import Dict

# A `setup_logging` function is provided for convenience, but one can use whatever logging one 
# wishes, or none. DEBUG level will also log the model graphs.
setup_logging(log_file=None, level="INFO")

# Parse the arguments, e.g. the model path, device, and conversion options
# This is provided for convenience, but one can also just pass in the arguments directly to the
# `benchmark_model` API.
args, device = parse_benchmark_args()

# Load the model
model = ...

config = {
    "batch_size": args.batch_size,
    "n_samples": args.n_samples,
}

# Benchmark the model
# We squeeze the data tensor's dimensions prior to feeding it in, as the batch size of the generated
# data loader is controlled via the `batch_size` argument.
results: Dict[str, Dict[str, float]] = benchmark_model(
   model, config, args.conversions, data=torch.randn(1, 3, 28, 28).squeeze()
)
```

One would then run this the same way as before. 

#### Full working example:
A full working example can be found in `examples/mnist/benchmark_random_tensor.py` where a random 
input tensor is used, and the model is not trained at all.
E.g.
```bash
cd examples/mnist
python benchmark_random_tensor.py --conversion 10,2 --batch-size 100 --n-samples 5000
```


## CUDA pathing
For a number of the conversion options, one needs to provide one's CUDA path as an environmental
variable. This can be fed in via the command line (as below), or added to a `.env` file. For the 
laytter, an example `.env.example` has been provided, this can be adjiusted if needed and renamed
to `.env`.

Example command:
```bash
CUDA_HOME='/usr/local/cuda' python benchmark_random_tensor.py --conversion 10,2 --batch-size 100 --n-samples 5000
```
