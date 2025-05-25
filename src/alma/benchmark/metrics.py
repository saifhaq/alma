
from pydantic import BaseModel, Field, validator
from enum import Enum
import torch
from typing import Optional

from alma.utils.device import setup_device

class BenchmarkError(BaseModel):
    traceback: Optional[str] = Field(
        default=None, description="The traceback of the error"
    )
    error: Optional[str] = Field(
        default=None, description="The error message"
    )


class BenchmarkStatus(Enum):
    SUCCESS = "success"
    ERROR = BenchmarkError


class BenchmarkMetrics(BaseModel):
    """
    Base class for benchmark metrics.
    """
    device: Optional[torch.device] = Field(
        default=None, description="Device for benchmarking."
    )
    total_elapsed_time: float = Field(
        description="Total elapsed time during the benchmark"
    )
    batch_size: int = Field(
        gt=0, description="Batch size for benchmarking."
    )
    status: BenchmarkStatus = Field(
        description="Whether the benchmark succeeded or failed"
    )
    data_dtype: Optional[torch.dtype] = Field(  # Specify the type hint correctly
        default=torch.float32, description="The data type of the input data."
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("data_dtype", pre=True)
    def validate_dtype(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Convert string representation to torch.dtype
            return getattr(torch, v)
        if isinstance(v, torch.dtype):
            return v
        raise ValueError(f"Invalid dtype: {v}")


class TorchModuleMetrics(BenchmarkMetrics):
    """
    Contains the metrics for a torch.nn.Module benchmark.
    """
    total_samples: int = Field(
        gt=0, description="Total samples processed"
    )
    throughput: float = Field(
        description="Throughput of processed samples"
    )


class TextGenerationPipelineMetrics(BenchmarkMetrics):
    """
    Contains the metrics for a HuggingFace TextGenerationPipeline benchmark.
    """

    total_prompts: int = Field(
        gt=0, description="Total prompts processed"
    )
    total_input_tokens: int = Field(
        gt=0, description="Total input tokens processed"
    )
    total_output_tokens: int = Field(
        gt=0, description="Total output tokens generated"
    )
    output_throughput: float = Field(
        description="Output token throughput"
    )
    input_throughput: float = Field(
        description="Intput token throughput"
    )
    request_rate: float = Field(
        description="Rate that the prompts are processed"
    )

