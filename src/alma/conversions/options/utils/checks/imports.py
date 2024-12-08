import torch


def check_openxla():
    """
    Check that the openXLA backend is supported.
    """
    # Check if 'openxla' backend is available
    if "openxla" not in torch._dynamo.list_backends():
        raise RuntimeError(
            "OpenXLA backend is not available. Please ensure OpenXLA is installed and properly configured."
        )

    try:
        import torch_xla
    except ImportError:
        raise RuntimeError(
            "The torch-xla package is not available. Please install torch-xla to use 'openxla' backend.\n"
            "For installation instructions: https://github.com/pytorch/xla"
        )


def check_onnxrt():
    """
    Check that the onnxrt backend is supported.
    """
    if not torch.onnx.is_onnxrt_backend_supported():
        # Make sure all dependencies are installed, see here for a discussion by the ONNX team:
        # https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/onnxrt.py
        raise RuntimeError(
            "Need to install all dependencies. See here for more details: https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/onnxrt.py"
        )


def check_tvm():
    """
    Check that the tvm backend is supported.
    """
    # See here for some discussion of TVM backend:
    # https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/9

    # Check if 'tvm' backend is available
    if "tvm" not in torch._dynamo.list_backends():
        raise RuntimeError(
            "TVM backend is not available. Please ensure TVM is installed and properly configured."
        )
    try:
        import torch_tvm

        torch_tvm.enable()
    except ImportError:
        raise RuntimeError(
            "The torch-tvm package is not available. Please install torch-tvm to use 'tvm' backend.\n"
        )


def check_tensort():
    """
    Check that the tensorRT backend is supported.
    """
    try:
        import torch_tensorrt
    except ImportError:
        raise RuntimeError(
            "Torch TensorRT backend is not available. Please ensure it is installed and properly configured."
        )
