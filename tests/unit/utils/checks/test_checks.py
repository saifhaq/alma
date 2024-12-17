from src.alma.utils.checks.config import check_config
from src.alma.utils.checks.data import check_data_or_dataloader
from src.alma.utils.checks.inputs import check_input_type
from src.alma.utils.checks.model import check_model


class TestBenchmark(unittest.TestCase):
    def test_input_checks_config(self):


