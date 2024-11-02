import torch
import torch.nn as nn
import torch.quantization as tq
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.quantize_pt2e import prepare_pt2e

from quantization.qconfigs import (
    fake_quant_act,
    fixed_0255,
    learnable_act,
    learnable_weights,
)

# from torch._export import export


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

    def fuse_layers(self):
        """
        Fuses the layers of the model, used in eager mode quantization.
        """
        fused_model = torch.quantization.fuse_modules(
            self,
            [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]],
            inplace=False,
        )
        return fused_model

    def fx_quantize(self):
        """
        Quantizes the model with FX graph tracing. Does not do PTQ or QAT, and
        just provides default quantization parameters.

        Returns a fx-graph traced quantized model.
        """
        # Define qconfigs
        qconfig_global = tq.QConfig(
            activation=learnable_act(range=2),
            weight=tq.default_fused_per_channel_wt_fake_quant,
        )

        # Assign qconfigs
        qconfig_mapping = QConfigMapping().set_global(qconfig_global)

        # We loop through the modules so that we can access the `out_channels` attribute
        for name, module in self.named_modules():
            if hasattr(module, "out_channels"):
                qconfig = tq.QConfig(
                    activation=learnable_act(range=2),
                    weight=learnable_weights(channels=module.out_channels),
                )
                qconfig_mapping.set_module_name(name, qconfig)
            # Idiot pytorch, why do you have `out_features` for Linear but not Conv2d?
            elif hasattr(module, "out_features"):
                qconfig = tq.QConfig(
                    activation=learnable_act(range=2),
                    weight=learnable_weights(channels=module.out_features),
                )
                qconfig_mapping.set_module_name(name, qconfig)

        # Do symbolic tracing and quantization
        example_inputs = (torch.randn(1, 3, 224, 224),)
        self.eval()
        fx_model = prepare_qat_fx(self, qconfig_mapping, example_inputs)

        # Prints the graph as a table
        print("\nGraph as a Table:\n")
        fx_model.graph.print_tabular()
        return fx_model

    def eager_quantize(self):
        """
        Quantizes the model with eager mode. Does not do PTQ or QAT, and
        just provides default quantization parameters.

        Returns a eager quantized model.
        """

        # Fuse the layers
        fused_model = self.fuse_layers()

        # We loop through the modules so that we can access the `out_channels` attribute
        for name, module in fused_model.named_modules():
            if hasattr(module, "out_channels"):
                qconfig = tq.QConfig(
                    activation=learnable_act(range=2),
                    weight=learnable_weights(channels=module.out_channels),
                )
            # Idiot pytorch, why do you have `out_features` for Linear but not Conv2d?
            elif hasattr(module, "out_features"):
                qconfig = tq.QConfig(
                    activation=learnable_act(range=2),
                    weight=learnable_weights(channels=module.out_features),
                )
            else:
                qconfig = tq.QConfig(
                    activation=learnable_act(range=2),
                    weight=tq.default_fused_per_channel_wt_fake_quant,
                )
            module.qconfig = qconfig

        qconfig = tq.QConfig(
            activation=fixed_0255, weight=tq.default_fused_per_channel_wt_fake_quant
        )
        fused_model.quant_input.qconfig = qconfig

        # Do eager mode quantization
        fused_model.train()
        quant_model = torch.ao.quantization.prepare_qat(fused_model, inplace=False)

        return quant_model

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
