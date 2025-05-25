import logging

from ..conversions.conversion_options import ConversionOption
from .benchmark_config import BenchmarkConfig
from ..utils.isinstance import is_instance_of

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def check_and_return_callable(model, config: BenchmarkConfig, conversion: ConversionOption):
    if not is_instance_of(model, config.model_type):
        logger.info(f"Initializing model inside {conversion.mode} benchmarking")
        model = model()

        # Check the type is as expected
        assert is_instance_of(model, config.model_type), f"The provided callable should return an object of type {config.model_type} for a {config.config_type} benchmark config type"
    return model
