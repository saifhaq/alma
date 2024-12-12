from typing import Dict, Literal


def get_compile_settings(backend: Literal[str] = "inductor-default") -> Dict[str, str]:
    """
    Get the compilation settings for each torch.dynamo backend choice.

    Inputs:
    - backend (Literal[str]): The backend to use for torch.compile. Currently supported options in
        PyTorch are given by torch._dynamo.list_backends():
        ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        However, we have also split 'inductor' into 'inductor-default', 'inductor-max-autotune', and
        'inductor-reduce-overhead'. See here for an explanation of each:
        https://pytorch.org/get-started/pytorch-2.0/#user-experience

    Outputs:
    - compile_settings (Dict[str, str]): The returned compilation settings.
    """
    match backend:
        case "inductor-default":
            compile_settings = {
                "mode": "default",
                "backend": "inductor",
            }
        case "inductor-reduce-overhead":
            compile_settings = {
                "mode": "reduce-overhead",
                "fullgraph": True,
                "backend": "inductor",
            }

        case "inductor-max-autotune":
            import torch._inductor.config

            torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CPP"
            torch._inductor.config.max_autotune = True
            # TRITON backend fails with conv2d layers during max-autotune.
            torch._inductor.config.max_autotune_conv_backends = "ATEN"
            compile_settings = {
                "mode": "max-autotune",
                "fullgraph": True,
                "backend": "inductor",
                # Options can also be fed in via a dict, see here for options: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
                # However, I found this to be slower than using the "preset" options, so I think feeding
                # in any options will prevent any tuning from happening. This is why the settings
                # are controlled via global settings above.
                # options: {},
            }

        case "cudagraphs" | "openxla" | "tvm" | "tensorrt":
            compile_settings = {
                "fullgraph": True,
                "backend": backend,
            }
        case "tensorrt":
            # There are mnay options to choose from, see here for some detials:
            # https://pytorch.org/TensorRT/user_guide/torch_compile.html
            # Also, see here for a discussion on design decisions:
            # https://github.com/pytorch/TensorRT/discussions/2475
            compile_settings = {
                "fullgraph": True,
                "backend": backend,
                "options": {
                    "truncate_long_and_double": True,
                    "precision": torch.half,
                    # "enable_experimental_decompositions": True,
                    "use_fast_partitioner": False,  # Slower to compile, but should give a better result, e.g. max-autotune.
                    # "min_block_size": 2,
                    # "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
                    "optimization_level": 5,  # Highest level
                    "use_python_runtime": False,
                },
            }

        case "onnxrt":
            # TODO: debug why CUDA does not work on this.
            # See here for the accepted options for ONNXRT backend:
            # https://github.com/pytorch/pytorch/blob/05c1f37188b1923ee3e48634cb1aaa3e976e2d0f/torch/onnx/_internal/onnxruntime.py#L689
            from torch.onnx._internal.onnxruntime import OrtBackendOptions

            # Configure the ORT backend options. See here for options:
            # https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
            ort_options = OrtBackendOptions(
                # Specify preferred execution providers in priority order
                preferred_execution_providers=[
                    # ("CUDAExecutionProvider", {"device_id": 0}),  # For GPU execution
                    "CPUExecutionProvider",  # Fallback to CPU
                ],
                # Automatically infer execution providers from input/output devices
                infer_execution_providers=True,
                # Optional: Set to True if you want to pre-allocate output memory
                preallocate_output=True,
                # Disable AOT autograd for training support
                use_aot_autograd=False,
            )
            compile_settings = {
                "fullgraph": True,
                "backend": backend,
                "options": ort_options,
            }
        case _:
            raise ValueError(f"{backend} is not a valid option")

    return compile_settings
