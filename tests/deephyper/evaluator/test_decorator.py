import unittest
import pytest

from deephyper.evaluator import profile


@profile
def run_profile(config):
    return config["x"]


@pytest.mark.fast
@pytest.mark.hps
class TestDecorator(unittest.TestCase):
    def test_profile(self):

        output = run_profile({"x": 0})

        assert "timestamp_end" in output["metadata"]
        assert "timestamp_start" in output["metadata"]
        assert 0 == output["objective"]
