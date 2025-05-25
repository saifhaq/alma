# MNIST example

This example of `alma` demonstrates how to train a simple model on the MNIST dataset and benchmark
the model speed for different conversions using the `benchmark_model` API. The example covers:

- Using custom dataloaders or auto-generated data
- Error handling and graceful failure modes
- CLI integration and argument parsing
- Multiprocessing for isolated benchmarking environments
- Comprehensive logging and debugging support
- Non-blocking data transfer optimization
- Device fallback handling

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

## Benchmark conversion options using a default data loader:

The `benchmark_model` API is used to benchmark the speed of a model for different conversion options.
The API is used as follows (see `benchmark_with_dataloader.py`):

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results

# Parse the arguments, e.g. the model path, device, and conversion options
# This is provided for convenience, but one can also just pass in the arguments directly to the
# `benchmark_model` API.
args, conversions = parse_benchmark_args()

# Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
model = ...

# Load the data
data_loader = YourDataLoader(
    dataset,
    batch_size=100,
    shuffle=False,
    num_workers=8, # Along with pinned memory, number of workers can optimize data load times
    pin_memory=True,
)

# Set the configuration
config = BenchmarkConfig(
    n_samples=args.n_samples,
    batch_size=args.batch_size,
    device=device,  # The device to run the model on
)

# Benchmark the model
results = benchmark_model(
   model, config, conversions, data_loader=data_loader
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

### Full working example:
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

## Pre-defined argparser for easy control and experimentation
In the above example, the batch size of the data loader is controlled via the `--batch_size` 
argument, fed in via the config object.
The number of samples to run the benchmark on  is controlled via the `--n_samples` argument. 
For convenience, we also provide a `--data-dir` argument, so that one can have one's data loader 
feed in specific data, and a `--model-path` argument, so that one can feed in specific model weights.

Finally, we also provide an `--ipdb` argument, which throws one into an ipdb debugging session if and 
wherever an Exception occurs. See this 
[blog post](https://medium.com/@oscar-savolainen/my-favourite-python-snippets-794d5653af38) for 
more details on the ipdb sysexception hook. This will not work if one sets the `fail_with_error` argument,
which is discussed later in section [Graceful or fast failure](#graceful-or-fast-failure), to False.

One can of course also just pass in non-CLI arguments directly to the `benchmark_model` API.


### Selecting conversion options:

To see all of the conversion options via the CLI `parse_benchmark_args` argparser function, run 
the following command:

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

## Implicitly initialise a data loader inside of `benchmark_model`:

One does not need to pass in a data loader to do the benchmarking. However, if one does not pass in a data loader,
one has to pass in a `data` input, where `data` is just a tensor with batch size 1.
The benchmarking will generate a data loader, where the size of the generated tensors is
taken from an inputted `data` tensor, which can be a tensor with random values. Either the `data`
tensor or the `data_loader` has to be provided. The batch size of the generated data loader is
controlled via the `batch_size` argument.

```python
from alma import benchmark_model
from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark import BenchmarkConfig
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
args, conversions = parse_benchmark_args()

# Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
model = ...

# Set the configuration
config = BenchmarkConfig(
    n_samples=args.n_samples,
    batch_size=args.batch_size,
    device=device,  # The device to run the model on
)

# Benchmark the model
# We squeeze the data tensor's dimensions prior to feeding it in, as the batch size of the generated
# data loader is controlled via the `batch_size` argument.
results: Dict[str, Dict[str, float]] = benchmark_model(
   model, config, conversions, data=torch.randn(1, 3, 28, 28).squeeze()
)

# Display the results
display_all_results(results)
```

One would then run this the same way as before. 

### Full working example:
A full working example can be found in `examples/mnist/benchmark_random_tensor.py` where a random 
input tensor is used, and the model is not trained at all.
E.g.
```bash
cd examples/mnist
python benchmark_random_tensor.py --conversion 10,2 --batch-size 100 --n-samples 5000
```


## Graceful or fast failure
By default, `alma` will fail fast if any conversion method fails. This is because we want to know
if a conversion method fails, so that we can fix it. 
However, if one wants to continue benchmarking other options even if a conversion method fails, 
one can set `fail_on_error` to False in the config dictionary.
`alma` will then fail gracefully for that method. One can then access the associated error messages 
and full tracebacks for the failed methods from the returned object.

This is useful for debugging and understanding why a given conversion failed, e.g. because of 
hardware incompatabilities, missing dependencies, etc. 

For example, if the `FAKE_QUANTIZED` conversion fails because it's not currently supported for Apple
silicon, one's code with `fail_on_error=False`, may look like this:

```python
...

# Set the configuration
config = BenchmarkConfig(
    n_samples=args.n_samples,
    batch_size=args.batch_size,
    device=device,  # The device to run the model on
    fail_on_error=False,  # If True, the benchmark will fail fast if a conversion method fails
)

conversions = ["EAGER", "NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC"]

# Benchmark the model
results: Dict[str, Dict[str, float]] = benchmark_model(
   model, config, conversions, data=data
)

# Display the results
display_all_results(results)
```

And the results may look like this (did not fail when the error occured):

```bash
EAGER results:
Device: mps
Total elapsed time: 0.0871 seconds
Total inference time (model only): 0.0085 seconds
Total samples: 2048 - Batch size: 64
Throughput: 241577.09 samples/second


NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC results:
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
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/examples/mnist/benchmark_random_tensor.py", line 60, in <module>
    main()
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/examples/mnist/benchmark_random_tensor.py", line 48, in main
    results: Dict[str, Dict[str, Any]] = benchmark_model(
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark_model.py", line 101, in benchmark_model
    result, stacktrace = benchmark_process_wrapper(
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/utils/processing.py", line 36, in run_benchmark_process
    result = benchmark_func(device, *args, **kwargs)
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark/benchmark.py", line 21, in benchmark
    benchmark(
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/utils/processing.py", line 144, in wrapper
    result: dict = func(*args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark/benchmark.py", line 81, in benchmark
    warmup(forward_call, data_loader, device)
  File "/Users/oscarsavolainen/Coding/Mine/Alma-Saif/src/alma/benchmark/warmup.py", line 33, in warmup
    _ = forward_call(data)
        ^^^^^^^^^^^^^^^^^^
  File "<eval_with_key>.2", line 5, in forward
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/torch/ao/quantization/_learnable_fake_quantize.py", line 160, in forward
    X = torch._fake_quantize_learnable_per_tensor_affine(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NotImplementedError: The operator 'aten::_fake_quantize_learnable_per_tensor_affine' is not 
currently implemented for the MPS device. If you want this op to be added in priority during the 
prototype phase of this feature, please comment on 
https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment 
variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this 
will be slower than running natively on MPS.
```

By default `display_all_results` only logs the error from the conversion, but one can also 
include the traceback in `display_all_results` via the `include_traceback_for_errors` argument
E.g. `display_all_results(results, include_traceback_for_errors=True)`.

If there are any issues with the traceback, please let us know! Stitching the traceback together
through multiprocessing and error-handler wrappers was non-trivial to do, but hopefully it gives 
informative tracebacks! If anything can be improved, please raise an issue!

## Isolated environments for each conversion method via multi-processing</summary>

By default, `alma` will run each conversion method in a separate process (one at a time), so that 
one can benchmark each conversion method in isolation. This ensures that each conversion method is benchmarked
in a fair and isolated environment, and is relevant because some of the methods (e.g. optimum quanto)
can affect the global torch state and break other methods (e.g. by overwriting tensor defaults in 
the C++ backend).

To disable this, one can set `multiprocessing` to False in the config dictionary.
E.g.

```python
# Set the configuration
config = BenchmarkConfig(
    n_samples=args.n_samples,
    batch_size=args.batch_size,
    device=device,  # The device to run the model on
    multiprocessing=False,  # If True, each conversion option will be run in a separate process.
    fail_on_error=False,  # If True, the benchmark will fail fast if a conversion method fails.
)
```



### Effect on model memory
A consequence of running in multiple processes is that the model, if initialized naively, will be copied
from the parent process to the child process. This doubles the required model memory, which can be 
a problem for large models. To avoid this, one can, insead of feeding in a model
directly to `benchmark_model`, feed in a function that returns the model. This way, the model is 
only initialized once the child process starts for each conversion method, meaning we only have 
one copy of the model at a time in device memory. 

To make this easy, we provide a `lazyload` decorator one can use at model initialisation to have
it only load once it is called.
E.g.

```python
from alma.utils.multiprocessing.lazyload import lazyload
...

# The model will only be properly initialised once called inside the benchmark process, meaning
# multiple copies of the model will not be made (here and inside the child process).
@lazyload
model = Net()

config = BenchmarkConfig(
    n_samples=args.n_samples,
    batch_size=args.batch_size,
    device=device,  # The device to run the model on
    multiprocessing=False,  # If True, each conversion option will be run in a separate process.
)

# Feed in `get_model` instead of the model directly
results: Dict[str, Dict[str, float]] = benchmark_model(
   get_model, config, conversions, data=data
)
```

If one does feed in a model directly, with `multiprocessing=True`, the program will throw a warning
but continue.

#### Full working example:

A full working example can be found in `examples/mnist/mem_efficient_benchmark_rand_tensor.py` where
a callable function is fed in that returns the model. 


## Using a dict for the config
One is not required to use `BenchmarkConfig` to set the configuration. In the spirit of reducing the
amount of code one needs to write, one can also just pass in a dictionary. E.g.

```python
config = {
    "n_samples": args.n_samples,
    "batch_size": args.batch_size,
    "device": device,  # The device to run the model on
    "multiprocessing": False,  # If True, each conversion option will be run in a separate process.
    "fail_on_error": False,  # If True, the benchmark will fail fast if a conversion method fails.
}
```
This will throw an error if any of the required keys are missing or if the types are incorrect.

## Device fallbacks
`BenchmarkConfig` exposes a few more options, related to device fallbacks. These are:
- allow_device_override
- allow_cuda
- allow_mps

`allow_device_override` is a boolean that defines whether or not we will allow `alma` to move 
conversion methods to specific devices, if the conversion method in question only works on that 
device. E.g. `ONNX_CPU` will fail on GPU, as will PyTorch's native converted quantized models 
which are CPU only: `NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED`. This is `True` by default, but it is 
very much up to the user. If you want the methods to fail if not compatiblewith `device`, then set 
this to `False`. If you want `alma` to automatically move the method to the appropriate device, 
leave it as `True`.

`allow_cuda` and `allow_mps` are guides on which device to fallback to in case `device` fails to 
run the conversion method in question. If `allow_cuda=True` and CUDA is available, then it will 
default to cuda. If not, then it will similarly check `mps`.

As such, a "complete" config looks like this:
```python

config = BenchmarkConfig(
    n_samples=args.n_samples,
    batch_size=args.batch_size,
    device=device,
    multiprocessing=True,
    fail_on_error=False,
    allow_device_override=True,  # Allow device override for device-specific conversions
    allow_cuda=True,  # True allows CUDA as an override option
    allow_mps=True,  # True allows MPS as an override option
)
```

## Non-blocking host to device data trasnfer
In many cases, speedups can be gained via setting the transfer of data from host (e.g. CPU) to 
device (e.g. GPU) to be `non_blocking`:
```python
cuda_tensor = cpu_tensor.to("cuda", non_blocking=True)
```

[This blog](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html) discusses it 
extensively, as well as the option of using pinned memory and multiple workers in the data loader. 
We follow the PyTorch convention of having it default to `False`, but provide it as an option in 
the config, one can set it to `True`.

```python
config = BenchmarkConfig(
    n_samples=args.n_samples,
    batch_size=args.batch_size,
    device=device,
    non_blocking=True,
)
```

## Logging and CI integration

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

