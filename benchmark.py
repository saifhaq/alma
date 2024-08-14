import argparse
import logging
import os
import time
from typing import Iterator, Tuple, Callable
import onnx

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

from train import Net


def setup_logging(log_file: str = None) -> None:
    """
    Sets up logging to print to console and optionally to a file.

    Args:
        log_file (str): Path to a log file. If None, logs will not be saved to a file.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ],
    )


class CustomImageDataset(Dataset):
    def __init__(self, img_dir: str, transform: transforms.Compose = None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = []
        self._gather_images(img_dir)

    def _gather_images(self, directory: str) -> None:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    self.img_paths.append(os.path.join(root, file))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.img_paths[idx % len(self.img_paths)]  # Circular indexing
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning 0 as label since it's not used for benchmarking


class CircularSampler(Sampler):
    def __init__(self, data_source: Dataset):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from torch.randperm(len(self.data_source)).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class CircularDataLoader(DataLoader):
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False, **kwargs
    ):
        sampler = CircularSampler(dataset)
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if model_path.endswith(".pt"):
        try:
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location=device)
            model = Net()  # Define your model architecture (Net)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
    else:
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.eval()
    return model

def get_compiled_model(model: torch.nn.Module, data_loader: DataLoader , device: torch.device):
    """
    Compile the model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device to run the model on

    Outputs:
    model (torch._dynamo.eval_frame.OptimizedModule): The compiled model

    """
    logging.info("Running torch.compile the model")            # Get a sample of data to pass through the model

    # See below for documentation on torch.compile and a discussion of modes
    # https://pytorch.org/get-started/pytorch-2.0/#user-experience
    compile_settings = {
        # 'mode': "reduce-overhead",  # Good for small models
        "mode": "max-autotune",  # Slow to compile, but should find the "best" option
        "fullgraph": True,  # Compiles entire program into 1 graph, but comes with restricted Python
    }

    model = torch.compile(model, **compile_settings)

    # Pass some data through the model to have it compile
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        _ = model(data)
        break

    return model

def get_exported_model(model, data_loader: DataLoader, device: torch.device):
    """
    Export the model using torch.export.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device to run the model on

    Outputs:
    model (torch.export.Model): The exported model 
    """

    logging.info("Running torch.export the model")            # Get a sample of data to pass through the model
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        break

    # Call torch export, which decomposes the forward pass of the model
    # into a graph of Aten primitive operators
    model = torch.export.export(model, (data,))
    model.graph.print_tabular()

    return model

def get_onnx_model(model, data_loader: DataLoader, device: torch.device):
    """
    Export the model to ONNX using torch.onnx.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device to run the model on

    Outputs:
    None
    """
    logging.info("Running torch.onnx the model")            # Get a sample of data to pass through the

    # Input to the model
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        break

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    # 
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = ["input_data"] + [name for name, _ in model.named_parameters()]
    output_names = ["output"]

    # torch.onnx.export(model, data, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)
    model.eval()

    # Export the model
    torch.onnx.export(model,               # model being run
        data,                         # model input (or a tuple for multiple inputs)
        "model.onnx",   # where to save the model (can be a file or file-like object)
        verbose=True,
        export_params=True,        # store the trained parameter weights inside the model file
        # opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = input_names,   # the model's input names
        output_names = output_names,  # the model's output names
        # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
        #             'output' : {0 : 'batch_size'}}
    )

    # Check the model is well formed
    # Load the ONNX model
    loaded_model = onnx.load("model.onnx")

    # Check that the model is well formed
    onnx.checker.check_model(loaded_model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(loaded_model.graph))


def benchmark_onnx(
    model_path: str, device: torch.device, data_loader: DataLoader, n_images: int, exported: bool = False
    ) -> None:
    """
    Benchmark an ONNX model using ONNX Runtime, however it is buggy and leads
    to a seg-fault on Oscar's Mac.
    """
    import onnxruntime
    total_time = 0.0
    total_images = 0
    num_batches = 0

    ort_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    start_time = time.time()  # Start timing for the entire process

    for data, _ in tqdm(data_loader, desc="Benchmarking"):
        if total_images >= n_images:
            break

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
        # data = data.to(device)
        batch_start_time = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        # output = model(data)
        batch_end_time = time.time()

        batch_size = min(data.size(0), n_images - total_images)
        total_time += batch_end_time - batch_start_time
        total_images += batch_size
        num_batches += 1

        if total_images >= n_images:
            break

    end_time = time.time()  # End timing for the entire process

    total_elapsed_time = end_time - start_time
    throughput = total_images / total_elapsed_time if total_elapsed_time > 0 else 0
    logging.info(f"Total elapsed time: {total_elapsed_time:.4f} seconds")
    logging.info(f"Total inference time (model only): {total_time:.4f} seconds")
    logging.info(f"Total images: {total_images}")
    logging.info(f"Throughput: {throughput:.2f} images/second")


def benchmark_model(
    model: torch.nn.Module, device: torch.device, data_loader: DataLoader, n_images: int, exported: bool = False
) -> None:
    total_time = 0.0
    total_images = 0
    num_batches = 0

    def get_forward_call_function(model, exported: bool) -> Callable:
        if exported:
            forward = model.module().forward
        # elif onnx:
        #
        else:
            forward = model.forward
        return forward

    forward_call = get_forward_call_function(model, exported)

    start_time = time.time()  # Start timing for the entire process

    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Benchmarking"):
            if total_images >= n_images:
                break

            data = data.to(device)
            batch_start_time = time.time()
            output = forward_call(data)
            # output = model(data)
            batch_end_time = time.time()

            batch_size = min(data.size(0), n_images - total_images)
            total_time += batch_end_time - batch_start_time
            total_images += batch_size
            num_batches += 1

            if total_images >= n_images:
                break

    end_time = time.time()  # End timing for the entire process

    total_elapsed_time = end_time - start_time
    throughput = total_images / total_elapsed_time if total_elapsed_time > 0 else 0
    logging.info(f"Total elapsed time: {total_elapsed_time:.4f} seconds")
    logging.info(f"Total inference time (model only): {total_time:.4f} seconds")
    logging.info(f"Total images: {total_images}")
    logging.info(f"Throughput: {throughput:.2f} images/second")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Models")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model file (e.g., mnist_cnn_quantized.pt, mnist_cnn_scripted.pt, mnist_cnn.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing images for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        metavar="N",
        help="input batch size for benchmarking (default: 30)",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=5000,
        help="Total number of images to process (default: 300)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA acceleration",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables MPS acceleration",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="run torch.compile on the model",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        default=False,
        help="run torch.export on the model",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        default=False,
        help="export the model to ONNX",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Ensure all images are resized to 28x28
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = CustomImageDataset(img_dir=args.data_dir, transform=transform)

    data_loader = CircularDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model_path, device)

    if args.compile:
        model = get_compiled_model(model, data_loader, device)

    if args.export:
        model = get_exported_model(model, data_loader, device)

    if args.onnx:
        model = get_onnx_model(model, data_loader, device)

    if args.onnx:
        benchmark_onnx('model.onnx', device, data_loader, args.n_images, args.export)
    else:
        benchmark_model(model, device, data_loader, args.n_images, args.export)


if __name__ == "__main__":
    setup_logging()
    main()
