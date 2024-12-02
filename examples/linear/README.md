# Simple Linear example
This example of `alma` demonstrates how to benchmark the model speed for different conversions 
using the `benchmark_model` API. No model training is done, it is just a simple linear layer and
ReLU activation. We do not provide a data loader for this example, we only show the API option
where one can feed in a random tensor to initialise the benchmarking data loader.

For an example where we feed in a dataloader into the `benchmark_model` API, see the `mnist` example.

## Benchmark with `alma`:


### Benchmark conversion options using a data tensor to intialise a data loader:

One does not need to pass in a data loader to do the benchmarking. If one does not pass in a data loader,
one has to pass in a `data` input, where `data` is just a tensor with batch size 1.
The benchmarking will generate a data loader, where the size of the generated tensors is
taken from an inputted `data` tensor, which can be random. Either the `data` tensor or the `data_loader`
has to be provided. The batch size of the generated data loader is controlled via the `batch_size` argument,
which is why we squeeze the data tensor's dimensions prior to feeding it in.

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args

# Parse the arguments, e.g. the model path, device, and conversion options
# This is provided for convenience, but one can also just pass in the arguments directly to the
# `benchmark_model` API.
args, device = parse_benchmark_args()

# Load the model
model = ...

# Benchmark the model
benchmark_model(
    model, device, args, args.conversions, data=torch.randn(1, 3, 28, 28).squeeze()
)
```

A working example can be found in `examples/mnist/benchmark_random_tensor.py` where a random model is used.
E.g.
```bash
cd examples/mnist
python YOUR_SCRIPT_NAME.py  --conversions EAGER,EXPORT+EAGER --batch-size 10
--n-samples 5000 
```

This will run the EXPORT+EAGER conversion option and the EAGER conversion option, benchmarking the
model speed for each conversion option. The batch size of the data loader is controlled via the
`batch_size` argument. The number of samples to run the benchmark on is controlled via the `n_samples`
argument. For convenience, we also provide a `data-dir` argument, so that one can have one's
data loader feed in specific data.

The results will look like this, depending on one's model, dataloader and hardware.

```bash
EAGER results:
Total elapsed time: 0.0565 seconds
Total inference time (model only): 0.0034 seconds
Total samples: 5000
Throughput: 88528.94 samples/second

EXPORT+EAGER results:
Total elapsed time: 0.0350 seconds
Total inference time (model only): 0.0026 seconds
Total samples: 5000
Throughput: 142958.75 samples/second
```

### Benchmarking all of the conversion options:
If one wants to test all of the conversion options, one can just not pass in the `--conversion` argument.
```bash
cd examples/linear
python benchmark_random_tensor.py --batch-size 100 --n-samples 5000
```

To see all of the conversion options, run the following command:
```bash
cd examples/linear
python benchmark_random_tensor.py --help
```


### Benchmarking specific conversion options:
The conversion options can be selected as either an integer or a string. The integer corresponds to the
index of the conversion option in the list of all conversion options. The string corresponds to the name
of the conversion option. These can be mixed and matched.

To feed in multiple conversion options, separate the conversion options with a comma. For example, one
can run:

```bash
cd examples/mnist
python benchmark_random_tensor.py  --conversions 2,EAGER
```

This will run the 2nd conversion option and the EAGER conversion option.


## CUDA pathing
For a number of the conversion options, one needs to provide one's CUDA path as an environmental
variable. This can be fed in via the command line (as below), or added to a `.env` file. For the 
laytter, an example `.env.example` has been provided, this can be adjiusted if needed and renamed
to `.env`.

Example command:
```bash
CUDA_HOME='/usr/local/cuda' python benchmark_random_tensor.py  --conversion 1,2 --n-samples 5000 --batch-size 100
```
