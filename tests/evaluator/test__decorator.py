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
    # To simulate to memory allocation
    x = [0]
    return x[0]


def test_profile():
    # Scalar output
    output = run_scalar_output({"x": 0})

    # output = {"output": initial_output, "metadata": {}}
    assert "output" in output
    assert output["output"] == 0
    assert "metadata" in output
    assert "timestamp_end" in output["metadata"]
    assert "timestamp_start" in output["metadata"]

    # Dict output without metadata
    output = run_dict_without_metadata_output({"x": 0})

    assert "output" in output
    assert isinstance(output["output"], dict)
    assert "objective" in output["output"]
    assert output["output"]["objective"] == 0
    assert "metadata" in output and "metadata" not in output["output"]
    assert "timestamp_end" in output["metadata"]
    assert "timestamp_start" in output["metadata"]

    # Dict output with None metadata
    output = run_dict_with_none_metadata_output({"x": 0})

    assert "output" in output
    assert isinstance(output["output"], dict)
    assert "objective" in output["output"]
    assert output["output"]["objective"] == 0
    assert "metadata" in output and "metadata" in output["output"]
    assert "timestamp_end" in output["metadata"]
    assert "timestamp_start" in output["metadata"]
    assert output["output"]["metadata"] is None

    # Dict output with empty dict metadata
    output = run_dict_with_empty_dict_metadata_output({"x": 0})

    assert "output" in output
    assert isinstance(output["output"], dict)
    assert "objective" in output["output"]
    assert output["output"]["objective"] == 0
    assert "metadata" in output and "metadata" in output["output"]
    assert "timestamp_end" in output["metadata"]
    assert "timestamp_start" in output["metadata"]
    assert isinstance(output["output"]["metadata"], dict) and len(output["output"]["metadata"]) == 0

    # Dict output with metadata
    output = run_dict_with_metadata_output({"x": 0})

    assert "output" in output
    assert isinstance(output["output"], dict)
    assert "objective" in output["output"]
    assert output["output"]["objective"] == 0
    assert "metadata" in output and "metadata" in output["output"]
    assert "timestamp_end" in output["metadata"]
    assert "timestamp_start" in output["metadata"]
    assert "foo" in output["output"]["metadata"]
    assert output["output"]["metadata"]["foo"] == 0

    # Scalar output with memory profiling
    output = run_scalar_output_with_memory({"x": 0})
    assert "output" in output
    assert output["output"] == 0
    assert "metadata" in output
    assert "timestamp_end" in output["metadata"]
    assert "timestamp_start" in output["metadata"]
    assert "memory" in output["metadata"]
    assert output["metadata"]["memory"] > 0
