import sys
import pandas as pd


def add_subparser(subparsers):
    subparser_name = "csv"
    function_to_call = main

    parser = subparsers.add_parser(
        subparser_name, help="Tool to analyse a results.csv file produced by the search."
    )
    subparsers = parser.add_subparsers(help="Kind of analytics.")

    # best search_spaces
    subparser = subparsers.add_parser(
        "best", help="print the best arch_seq"
    )
    subparser.add_argument("path", type=str, help="path to the CSV file.")

    return subparser_name, function_to_call


def main(*args, **kwargs):

    if sys.argv[2] == "best":

        df = pd.read_csv(sys.argv[3])
        i = df.objective.argmax()
        row = df.iloc[i].tolist()
        arch_seq = row[:-2]

        if all([el%1 == 0 for el in arch_seq]):
            arch_seq = [int(el) for el in arch_seq]

        objective = row[-2]
        print("Objective: ", objective)
        print("Arch Seq: ", arch_seq)