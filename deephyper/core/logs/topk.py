"""
A command line to extract the top-k best configuration from a DeepHyper execution::

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
    subparser_name = "topk"
    function_to_call = main

    parser = subparsers.add_parser(subparser_name, help="Print the top-k configurations.")

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


def output_best_configuration(path: str, output: str, k: int, **kwargs) -> None:
    """Output the configuration based on the maximal objective found in the CSV input file.

    Args:
        path (str): Path of the CSV input file.
        output (str): Path of the output file ending in (.csv|.yaml|.json).
        k (int): Number of configuration to output.
    """

    input_extension = path.split(".")[-1]
    if input_extension == "csv":
        df = pd.read_csv(path)
        df = df.sort_values(by=["objective"], ascending=False, ignore_index=True)
        subdf = df.iloc[:k]

        # if not ("arch_seq" in subdf.columns):

        #     try:
        #         if (subdf.to_numpy()[:, :-3] < 1).all():
        #             conv_type = float
        #         else:
        #             conv_type = int

        #         subdf = pd.DataFrame(
        #             {
        #                 "arch_seq": [
        #                     str(list(el)) for el in subdf.to_numpy()[:, :-3].astype(conv_type)
        #                 ],
        #                 "objective": subdf.objective.tolist(),
        #                 "elapsed_sec": subdf.elapsed_sec.tolist(),
        #                 "duration": subdf.duration.tolist()
        #             }
        #         )
        #     except TypeError: pass

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
    else:
        raise DeephyperRuntimeError(
            f"The specified input file extension '{input_extension}' is not supported."
        )


def main(*args, **kwargs):

    output_best_configuration(**kwargs)
