"""
Top-K Configuration
-------------------

A command line to extract the top-k best configuration from a DeepHyper execution.

It can be used with:

.. code-block:: console

    $ deephyper-analytics --help
    usage: deephyper-analytics topk [-h] [-k K] [-o OUTPUT] path

    positional arguments:
    path                  Path to the input CSV file.

    optional arguments:
    -h, --help            show this help message and exit
    -k K                  Number of best configurations to output in decreasing order of best objective.
    -o OUTPUT, --output OUTPUT
                            Path to the output file.

An example usage is:

.. code-block:: console

    $ deephyper-analytics topk combo_8gpu_8_agebo/infos/results.csv -k 2
    '0':
    arch_seq: '[229, 0, 22, 1, 1, 53, 29, 1, 119, 1, 0, 116, 123, 1, 273, 0, 1, 388]'
    batch_size: 59
    elapsed_sec: 10259.2741303444
    learning_rate: 0.0001614947
    loss: log_cosh
    objective: 0.9236862659
    optimizer: adam
    patience_EarlyStopping: 22
    patience_ReduceLROnPlateau: 10
    '1':
    arch_seq: '[229, 0, 22, 0, 1, 235, 29, 1, 313, 1, 0, 116, 123, 1, 37, 0, 1, 388]'
    batch_size: 51
    elapsed_sec: 8818.2674164772
    learning_rate: 0.0001265946
    loss: mae
    objective: 0.9231553674
    optimizer: nadam
    patience_EarlyStopping: 23
    patience_ReduceLROnPlateau: 14


An ``--output`` argument is also available to save the output in a YAML, JSON or CSV format.
"""
import json
import pandas as pd
import yaml

from deephyper.core.exceptions import DeephyperRuntimeError


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "topk"
    function_to_call = main

    parser = subparsers.add_parser(
        subparser_name, help="Print the top-k configurations."
    )

    # best search_spaces
    parser.add_argument("path", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "-k",
        type=int,
        default=1,
        required=False,
        help="Number of best configurations to output in decreasing order of best objective.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="",
        help="Path to the output file.",
    )

    return subparser_name, function_to_call


def output_best_configuration_from_csv(
    path: str, output: str, k: int, **kwargs
) -> None:
    """Output the configuration based on the maximal objective found in the CSV input file.

    :meta private:

    Args:
        path (str): Path of the CSV input file.
        output (str): Path of the output file ending in (.csv|.yaml|.json).
        k (int): Number of configuration to output.
    """

    input_extension = path.split(".")[-1]
    if input_extension == "csv":
        df = pd.read_csv(path)
        output_best_configuration_from_df(df, output, k)
    else:
        raise DeephyperRuntimeError(
            f"The specified input file extension '{input_extension}' is not supported."
        )


def output_best_configuration_from_df(df: str, output: str, k: int, **kwargs) -> None:
    """Output the configuration based on the maximal objective found in the CSV input file.

    :meta private:

    Args:
        df (DataFrame): a Pandas DataFrame.
        output (str): Path of the output file ending in (.csv|.yaml|.json).
        k (int): Number of configuration to output.
    """

    df = df.sort_values(by=["objective"], ascending=False, ignore_index=True)
    subdf = df.iloc[:k]

    if len(output) == 0:
        print(yaml.dump(json.loads(subdf.to_json(orient="index"))))
    else:
        output_extension = output.split(".")[-1]
        if output_extension == "yaml":
            with open(output, "w") as f:
                yaml.dump(json.loads(subdf.to_json(orient="index")), f)
        elif output_extension == "csv":
            subdf.to_csv(output)
        elif output_extension == "json":
            subdf.to_json(output, orient="index")
        else:
            raise DeephyperRuntimeError(
                f"The specified output extension is not supported: {output_extension}"
            )


def main(*args, **kwargs):
    """
    :meta private:
    """

    output_best_configuration_from_csv(**kwargs)
