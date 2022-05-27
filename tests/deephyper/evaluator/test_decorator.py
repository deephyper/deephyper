import unittest
import pytest

from deephyper.evaluator import profile


@profile
def run_profile(config):
    return config["x"]


class TestDecorator(unittest.TestCase):
    def test_profile(self):

        y = run_profile({"x": 0})

        assert "timestamp_end" in y
        assert "timestamp_start" in y
        assert y["objective"] == 0
