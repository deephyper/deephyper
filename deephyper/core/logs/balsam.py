"""Tools to extract information from Balsam jobs. An example commande line is::

    deephyper-analytics balsam workload --workflow test

This command will generate an output file name ``balsam-workload.json`` containing information about the number of jobs running in parallel at a given time.
"""
import json

from deephyper.core.exceptions import DeephyperRuntimeError

def add_subparser(subparsers):
    subparser_name = "balsam"
    function_to_call = main

    parser = subparsers.add_parser(
        subparser_name, help="Extract information from Balsam jobs."
    )

    # best search_spaces
    parser.add_argument("type", type=str, choices=["workload"])
    parser.add_argument("-w", "--workflow", type=str, help="Balsam workflow.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="",
        help="Path to the output file.",
    )

    return subparser_name, function_to_call


def extract_workload(workflow, output_file):

    from balsam.core.models import BalsamJob, process_job_times, utilization_report

    qs = BalsamJob.objects.filter(workflow=workflow)
    time_data = process_job_times(qs)
    times, num_running = utilization_report(time_data)
    times = [str(t) for t in times]
    num_running = [int(n) for n in num_running]

    data = {
        "timestamps": times,
        "num_jobs_running": num_running
    }

    if len(output_file) == 0:
        output_file = "balsam-workload.json"

    with open(output_file, "w") as f:
        json.dump(data, f)


def main(type, workflow, output, *args, **kwargs):

    if type == "workload":
        extract_workload(workflow, output)
    else:
        raise DeephyperRuntimeError(f"Analysis of type '{type}' is not accepted for balsam.")