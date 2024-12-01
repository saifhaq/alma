import logging
import time

import torch
import torch.fx as fx
import torch.nn.functional as F
import torch.optim as optim
from quantization.quantize import quantize_model
from quantization.utils import save_fake_quantized_model
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from arguments.train_args import parse_train_args
from data.datasets import CustomImageDataset
from data.transforms import TrainTransform
from model.model import Net
from utils.ipdb_hook import ipdb_sys_excepthook
from utils.setup_logging import setup_logging

# Adds ipdb breakpoint if and where we have an error
ipdb_sys_excepthook()


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
            start_time - end_time,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    args, device, train_kwargs, test_kwargs = parse_train_args()

    # Set up the train and test dataset and dataloaders
    train_dataset = CustomImageDataset(
        annotations_file="mnist_images.csv",
        img_dir="mnist_images",
        transform=TrainTransform,
    )
    # NOTE: the test and train data is the same.
    test_dataset = CustomImageDataset(
        annotations_file="mnist_images.csv",
        img_dir="mnist_images",
        transform=TrainTransform,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Set up the model, optimizer, and scheduler
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Train the model
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.quantize:
        quantize_model(args, model, device, train_loader, test_loader, train, test)
        if args.save_model:
            save_fake_quantized_model(model, "mnist_cnn_quantized.pt")

    elif args.save_model:
        # Save the "vanilla" model
        assert type(model) is not fx.GraphModule
        assert isinstance(model, torch.nn.Module)
        vanilla_file = "mnist_cnn.pt"
        torch.save(model.state_dict(), vanilla_file)
        logging.info(f"Floating point model saved to {vanilla_file}")

        # Save the scripted model
        scripted_file = "mnist_cnn_scripted.pt"
        model.save_scripted_model(scripted_file)
        logging.info(f"Scripted model saved to {scripted_file}")


if __name__ == "__main__":
    setup_logging()
    main()
