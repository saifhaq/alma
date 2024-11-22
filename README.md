# alma

Our customers have large amounts of images with digits that they want to classify using a ML model. The team has prepared the `setup-data.sh` and `train.py` scripts to come up with a classification model for digits. Have a look at that code and:
1. Train a model and save it to file;
2. Write a new Python script that loads the trained model and, given the `data_for_inference` folder, inspects all sub-folders and files and outputs the count of highest-likelihood predictions across all files - essentially how many occurrences of each digit are found in the target folder and subfolders.

Your inference script should scale to hundreds of thousands of files, and should be usable as part as a larger codebase. There is no need to worry about the results being "pretty", they only need to be human-readable, e.g.

```
$ python your_script.py --model mnist_cnn.pt --target data_for_inference
digit,count
0,29
4,1
```

You're free to change `setup-data.sh` and `train.py`, namely the data split and training setup, just document your reasoning. For convenience we include a `reset.sh` that you shouldn't need to change.

## Benchmark

We have the option to benchmark over a dozen different model conversion options. A full list of 
conversion options can be seen in the conversion field by the command:
```bash
python benchmark.py --help
```

To benchmark one's model, one needs to provide the path to the weights, a path to the data, and 
the desired conversion option (integers are used to select conversion options, for ease of use).

For a number of the conversion options, one needs to provide one's CUDA path as an environmental
variable. This can be fed in via the command line (as below), or added to a `.env` file. For the 
laytter, an example `.env.example` has been provided, this can be adjiusted if needed and renamed
to `.env`.

Example command:
```bash
CUDA_HOME='/usr/local/cuda' python benchmark.py --model path mnist_cnn.pt --data-dir 
data_for_inference --conversion 1 
```

## Code Standards
- **Black**: Ensures consistency following a strict subset of PEP 8.
- **isort**: Organizes Python imports systematically.

### Automatic Formatting Before Commit
1. From the repo root, run:
```bash
pre-commit install
```
### Manually Running Hooks
If you want to manually run all hooks on all files, you can do:

```bash
git stage .
pre-commit run --all-files
```



To quantize the model running on Apple silicon, run:
```
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py  --quantize
```
