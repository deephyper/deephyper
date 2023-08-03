import unittest
import pytest

from deephyper.evaluator import profile


@profile
def run_scalar_output(job):
    return 0


@profile
def run_dict_without_metadata_output(job):
    return {"objective": 0}


@profile
def run_dict_with_none_metadata_output(job):
    return {"objective": 0, "metadata": None}


@profile
def run_dict_with_empty_dict_metadata_output(job):
    return {"objective": 0, "metadata": {}}


@profile
def run_dict_with_metadata_output(job):
    return {"objective": 0, "metadata": {"foo": 0}}


@profile(memory=True)
def run_scalar_output_with_memory(job):
    return 0


@pytest.mark.fast
@pytest.mark.hps
class TestDecorator(unittest.TestCase):
    def test_profile(self):
        # Scalar output
        output = run_scalar_output({"x": 0})

        assert "objective" in output
        assert 0 == output["objective"]
        assert "metadata" in output
        assert "timestamp_end" in output["metadata"]
        assert "timestamp_start" in output["metadata"]

        # Dict output without metadata
        output = run_dict_without_metadata_output({"x": 0})

        assert "objective" in output
        assert 0 == output["objective"]
        assert "metadata" in output
        assert "timestamp_end" in output["metadata"]
        assert "timestamp_start" in output["metadata"]

        # Dict output with None metadata
        output = run_dict_with_none_metadata_output({"x": 0})

        assert "objective" in output
        assert 0 == output["objective"]
        assert "metadata" in output
        assert "timestamp_end" in output["metadata"]
        assert "timestamp_start" in output["metadata"]

        # Dict output with empty dict metadata
        output = run_dict_with_empty_dict_metadata_output({"x": 0})

        assert "objective" in output
        assert 0 == output["objective"]
        assert "metadata" in output
        assert "timestamp_end" in output["metadata"]
        assert "timestamp_start" in output["metadata"]

        # Dict output with metadata
        output = run_dict_with_metadata_output({"x": 0})

        assert "objective" in output
        assert 0 == output["objective"]
        assert "metadata" in output
        assert "timestamp_end" in output["metadata"]
        assert "timestamp_start" in output["metadata"]
        assert "foo" in output["metadata"]
        assert 0 == output["metadata"]["foo"]

        # Scalar output with memory profiling
        output = run_scalar_output_with_memory({"x": 0})

        assert "objective" in output
        assert 0 == output["objective"]
        assert "metadata" in output
        assert "timestamp_end" in output["metadata"]
        assert "timestamp_start" in output["metadata"]
        assert "memory" in output["metadata"]
        assert output["metadata"]["memory"] == 8


if __name__ == "__main__":
    test = TestDecorator()
    test.test_profile()
