# MNIST example

This example of `alma` demonstrates how to train a simple model on the MNIST dataset and benchmark
the model speed for different conversions using the `benchmark_model` API. Everything, from not using
a dataloader, to error handling, to CLI integration, to logging and debugging, is covered.

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
from alma.benchmark.log import display_all_results

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
results = benchmark_model(
   model, config, args.conversions, data_loader=data_loader
)

# Display the results
display_all_results(results)
```

One would run this script (e.g., if one wanted to benchmark the same model as used in training)
with the following commands (all steps from loading data, to training, to benchmarking):

```bash
python ABOVE_SCRIPT.py --model-path ./model/mnist.pt --conversions EAGER,EXPORT+EAGER --batch-size 10
--n-samples 5000 --data-dir data/data_for_inference
```

This will run the EXPORT+EAGER conversion option and the EAGER conversion option, benchmarking the
model speed for each conversion option. 

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

#### Argparsing:
In the above example, the batch size of the data loader is controlled via the `--batch_size` 
argument, fed in via the config object.
The number of samples to run the benchmark on  is controlled via the `--n_samples` argument. 
For convenience, we also provide a `--data-dir` argument, so that one can have one's data loader 
feed in specific data, and a `--model-path` argument, so that one can feed in specific model weights.

Finally, we also provide an `--ipdb` argument, which throws one into an ipdb debugging session if and 
wherever an Exception occurs. See this 
[blog post](https://medium.com/@oscar-savolainen/my-favourite-python-snippets-794d5653af38) for 
more details on the ipdb sysexception hook.

One can of course also just pass in non-CLI arguments directly to the `benchmark_model` API.


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
from alma.benchmark.log import display_all_results
from alma.utils.setup_logging import setup_logging
from typing import Dict

# Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
# as the model graphs. A `setup_logging` function is provided for convenience, but one can use 
# whatever logging one wishes, or none.
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

# Display the results
display_all_results(results)
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


### Error handling

In case where the benchmarking of a given conversion fails, it will return a dict for that conversion
which contains the error message as well as the full traceback. This is useful for debugging and
understanding why a given conversion failed, e.g. because of hardware incompatabilities, missing
dependencies, etc. 

For example, if the `FAKE_QUANTIZED` conversion fails because it's not currently supported for Apple
silicon, the default results may look like this:

```bash
CONVERT_QUANTIZED results:
Device: cpu
Total elapsed time: 0.8287 seconds
Total inference time (model only): 0.4822 seconds
Total samples: 5000 - Batch size: 50
Throughput: 6033.89 samples/second


FAKE_QUANTIZED results:
Benchmarking failed, error: The operator 'aten::_fake_quantize_learnable_per_tensor_affine' is not 
currently implemented for the MPS device. If you want this op to be added in priority during the 
prototype phase of this feature, please comment on 
https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment 
variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this 
will be slower than running natively on MPS.
```

The traceback of the error is also stored in the results dict, and can be accessed via 
`results[CONVERSION_NAME]["traceback"]`.

In this example, `print(results["FAKE_QUANTIZED"]["traceback"])` gives us:

```bash
Traceback (most recent call last):
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark_model.py", line 88, in benchmark_model
    result: Dict[str, float] = benchmark(
                               ^^^^^^^^^^
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark/benchmark.py", line 63, in benchmark
    warmup(forward_call, data_loader, device)
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark/warmup.py", line 26, in warmup
    _ = forward_call(data)
        ^^^^^^^^^^^^^^^^^^
  File "<eval_with_key>.173", line 5, in forward
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/quantization-tuts/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/quantization-tuts/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/quantization-tuts/lib/python3.11/site-packages/torch/ao/quantization/_learnable_fake_quantize.py", line 160, in forward
    X = torch._fake_quantize_learnable_per_tensor_affine(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NotImplementedError: The operator 'aten::_fake_quantize_learnable_per_tensor_affine' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
```

By default `display_all_results` only logs the error from the conversion, but one can also 
include the traceback in `display_all_results` via the `include_traceback_for_errors` argument
E.g. `display_all_results(results, include_traceback_for_errors=True)`.

Incidentally, this Apple-silicon-quantization issue can be solved by setting the environmental variable:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Logging and CI integration

A lot of the imported modules have verbose logging, and so `alma` provides some logging functions
to help manage the logging level. The `setup_logging` function is provided for convenience, but one
can use whatever logging one wishes, or none.

A lot of the conversion methods have extremely verbose logging. We have opted to wrap most of them
in a `suppress_output` context manager that silences all `sys.stdout` and `sys.stderr`. However, if one
sets ones logging level to DEBUG with the `setup_logging` function, then those internal import logs
will not be supressed.

Furthermore, as we have highlighted prior, we provide a `display_all_results` function to print 
the results in a nice format.There is also a `save_dict_to_json` function to save the results to a 
JSON file for easy CI integration.

### Debugging
If one is debugging, it is highly recommended that one use the `setup_logging` function and set one's
level to DEBUG. This will, among other things, log any torch.compile warnings and errors thrown by
torch.inductor that can point to issues in triton kernels, give verbose ONNX logging, and print 
the model graphs where appropriate.

