from typing import Any

from pydantic import Field, field_validator, BaseModel

from ..base import BenchmarkConfig

class TextGenerationPipelineKwargs(BaseModel):
    max_new_tokens: int = Field(
        default=128, gt=4, description="Maximum generated tokens for an input prompt."
    )
    min_new_tokens: int = Field(
        default=56, gt=0, description="Minimum generated tokens for an input prompt."
    )
    num_return_sequences: int = Field(
        default=1, gt=0, description="Number of generated sequences per input prompt."
    )

    @field_validator("min_new_tokens")
    def float_targets_validation(
        cls, min_new_tokens: int, info
    ) -> int:
        max_new_tokens = info.data.get("max_new_tokens")
        assert min_new_tokens < max_new_tokens, "`min_new_tokens` should be less than `max_new_tokens`"
        return min_new_tokens 

class TextGenerationPipelineBenchmarkConfig(BenchmarkConfig):
    """
    Configuration for benchmarking a `TextGenerationPipeline` object from HuggingFace
    transformers library. 
    """

    # Nested Pydantic model for pipeline kwargs
    pipeline_kwargs: TextGenerationPipelineKwargs = Field(
        default_factory=TextGenerationPipelineKwargs,
        description="Text generation pipeline specific parameters."
    )
    
    def get_kwargs_dict(self) -> dict[str, Any]:
        """Convert the pipeline_kwargs to a dictionary for use with the pipeline."""
        return self.pipeline_kwargs.model_dump()

