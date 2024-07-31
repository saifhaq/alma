import torch
import torch.quantization as tq
from torch.ao.quantization.fake_quantize import FakeQuantize, FixedQParamsFakeQuantize
from torch.ao.quantization._learnable_fake_quantize import (
    _LearnableFakeQuantize as LearnableFakeQuantize,
)


# observer: tq.HistogramObserver is used to collect statistics on the activations during training to determine the optimal quantization parameters.
# quant_min and quant_max: These define the range of quantized values (0 to 255 for quint8).
# dtype: The data type for quantization (torch.quint8 for unsigned 8-bit integers).
# qscheme: The quantization scheme (torch.per_tensor_affine), which means a single scale and zero-point for the entire tensor.
# scale: This is set to range / 255.0, where range is a parameter passed to the lambda function. This determines the scaling factor for quantization.
# zero_point: Set to 0.0, indicating no offset.
# use_grad_scaling: If True, it allows the gradients to scale appropriately during backpropagation, making the quantization parameters learnable.

learnable_act = lambda range : LearnableFakeQuantize.with_args(
    observer=tq.HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    scale=range / 255.0,
    zero_point=0.0,
    use_grad_scaling=True,
)

fixed_0255 = FixedQParamsFakeQuantize.with_args(
    observer=torch.ao.quantization.observer.FixedQParamsObserver.with_args(
        scale=1.0 / 255.0,
        zero_point=0.0,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )
)


learnable_weights = lambda channels : LearnableFakeQuantize.with_args(  # need to specify number of channels here
    observer=tq.PerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    scale=0.1,
    zero_point=0.0,
    use_grad_scaling=True,
    channel_len=channels,
)

fake_quant_act = FakeQuantize.with_args(
    observer=tq.HistogramObserver.with_args(
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
)