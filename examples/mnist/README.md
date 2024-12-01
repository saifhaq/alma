# Examples

Get data:
```bash
cd examples/mnist/data
./setup-data.sh
```

## Training model:

```bash
cd examples/mnist
python train.py --save-path ./model/mnist.pt
```

# Benchmark the model speed for different conversion options:
```bash
cd examples/mnist
python benchmark.py --model-path ./model/mnist.pt
```

See the benchmark script for more details on how to customize the benchmarking process to a given
model and data.

