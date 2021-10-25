"""
"""
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.core.plot import hps, multi, post_train, nas_single


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "notebook"
    function_to_call = main

    parser = subparsers.add_parser(
        subparser_name, help="Generate a notebook with different types of analysis"
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        choices = ["hps","nas"],
        help="Type of notebook to generate.",
    )
    parser.add_argument("path", nargs="+", type=str)
    parser.add_argument("-o", "--output", default="", type=str, required=False)

    return subparser_name, function_to_call


def notebook_for_hps(path: list, output: str) -> None:
    output_file = "dh-analytics-hps" if len(output) == 0 else output

    if len(path) == 1:
        hps.hps_analytics(path[0], output_file)
    else:
        raise DeephyperRuntimeError("Comparative analytics for HPS is not available yet!") # TODO

def notebook_for_nas(path: list, output: str) -> None:
    output_file = "dh-analytics-nas" if len(output) == 0 else output

    if len(path) == 1:
        nas_single.single_analytics(path[0], output_file)
    else:
        multi.multi_analytics(path, output_file)


def notebook_for_posttrain(path: list, output: str) -> None:
    output_file = "dh-analytics-posttrain" if len(output) == 0 else output

    if len(path) == 1:
        post_train.post_train_analytics(path[0], output_file)
    else:
        raise DeephyperRuntimeError("Comparative analytics for Post-Train is not available yet!") # TODO


def main(type: str, path: list, *args, **kwargs) -> None:
    """
    :meta private:
    """

    if type == "hps":
        notebook_for_hps(path, **kwargs)
    elif type == "nas":
        notebook_for_nas(path, **kwargs)
    elif type == "posttrain":
        notebook_for_posttrain(path, **kwargs)
    else:
        raise DeephyperRuntimeError(f"The notebook TYPE '{type}' passed is not supported.")
