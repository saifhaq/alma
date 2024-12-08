import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.quant_input = tq.QuantStub()  # used in eager mode
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # self.dequant_output = tq.DeQuantStub()  # used in eager mode

    def forward(self, x):
        # x = self.quant_input(x)  # used in eager mode
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2d(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.log_softmax(x)
        # output = self.dequant_output(output)  # used in eager mode
        return output

    def save_scripted_model(self, path: str) -> None:
        """
        Saves the model as a TorchScript model.

        Inputs:
        - path (str): The path to save the model to.

        Outputs:
        None
        """
        self.eval()
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
