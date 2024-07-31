import argparse
import logging
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

from typing import Tuple, Any, Union, List
from copy import deepcopy

import torch
import torch.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping

from qconfigs import learnable_act, learnable_weights, fake_quant_act
from ipdb_hook import ipdb_sys_excepthook

# Adds ipdb breakpoint if and where we have an error
ipdb_sys_excepthook()



from torch.ao.quantization.quantize_pt2e import prepare_pt2e
# from torch._export import export
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def setup_logging(log_file=None):
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
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        data = pd.read_csv(annotations_file)
        self.img_labels = data["target"]
        self.imgs = data["file"]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs.iloc[idx])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


    def quantize(self):
        """
        Quantizes the model with FX graph tracing. Does not do PTQ or QAT, and
        just provides default quantization parameters.
        """
        # Define qconfigs
        qconfig_global = tq.QConfig(
            activation=learnable_act(range=2),
            weight=tq.default_fused_per_channel_wt_fake_quant
        )

        # Assign qconfigs
        qconfig_mapping = QConfigMapping().set_global(qconfig_global)

        # We loop through the modules so that we can access the `out_channels` attribute
        for name, module in self.named_modules():
            if hasattr(module, 'out_channels'):
                qconfig = tq.QConfig(
                    activation=learnable_act(range=2),
                    weight=learnable_weights(channels=module.out_channels)
                )
                qconfig_mapping.set_module_name(name, qconfig)

        # Do symbolic tracing and quantization
        example_inputs = (torch.randn(1, 3, 224, 224),)
        self.eval()
        self = prepare_qat_fx(self, qconfig_mapping, example_inputs)

        # Prints the graph as a table
        print("\nGraph as a Table:\n")
        self.graph.print_tabular()
        
        return self 
        
    def save_scripted_model(self, path):
        self.eval()
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
        logging.info(f"Scripted model saved to {path}")

    def save_fake_quantized_model(self, path):
        # Ensure the model is in evaluation mode
        self.eval()
        # Save the quantized model state dict
        torch.save(self.state_dict(), path)
        logging.info(f"Quantized model saved to {path}")



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(
        tqdm(train_loader, desc=f"Training Epoch {epoch}")
    ):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
 
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    end_time = time.time()
    test_loss /= len(test_loader.dataset)

    logging.info(
        "\nTest set: Time: {:.2f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            start_time-end_time,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Training")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
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
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="quantize the model",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomInvert(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    train_dataset = CustomImageDataset(
        annotations_file="mnist_images.csv", img_dir="mnist_images", transform=transform
    )
    test_dataset = CustomImageDataset(
        annotations_file="mnist_images.csv", img_dir="mnist_images", transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.quantize:
        # Traces the model and quantizes
        model.quantize()
        # TODO: still need to do PTQ/QAT

    if args.quantize and args.save_model:
        model.save_fake_quantized_model('mnist_cnn_quantized.pt')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        model.save_scripted_model('mnist_cnn_scripted.pt')



if __name__ == "__main__":
    setup_logging()
    main()
