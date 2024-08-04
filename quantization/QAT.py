import logging
import time

import torch.optim as optim
import torch.quantization as tq
from torch.optim.lr_scheduler import StepLR


def QAT(train, test, args, model, device, train_loader, test_loader):
    """
    Perform QAT on the model using the given dataloaders, and train adn test
    functions.
    """

    # Reset the optimizer with included quantization parameters
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # NOTE: we may want seperate gamma, epochs and step_size for QAT
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Ensure fake quantization enabled
    model.apply(tq.enable_fake_quant)
    model.eval()

    # Disable PTQ observers
    for module in model.modules():
        if hasattr(module, "observer_enabled") or hasattr(module, "static_enabled"):
            module.disable_observer()

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    end_time = time.time()
    logging.info(f"QAT time: {end_time - start_time:.2f}s")
