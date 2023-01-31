import parse


def parse_result(stream: str) -> float:
    """Parse the output of a DeepHyper test. The format of the parsed output should be as follows:

    .. code-block::

        DEEPHYPER-OUTPUT: <float>

    Args:
        stream (str): The output of a DeepHyper test.
    Returns:
        float: The parsed output.
    """
    res = parse.search("DEEPHYPER-OUTPUT: {:f}", stream)
    return res[0]
