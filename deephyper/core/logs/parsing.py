import argparse
import datetime
import json
import os
import sys
import re
from shutil import copyfile

try:
    from balsam.core.models import BalsamJob, process_job_times, utilization_report

    if os.environ.get("BALSAM_SPHINX_DOC_BUILD_ONLY") == "TRUE":
        BALSAM_EXIST = False
        print("Module: 'balsam' module was found but not connected to a databse.")
    else:
        BALSAM_EXIST = True
        print("Module: 'balsam' module was found and connected to a databse.")
except ModuleNotFoundError as err:
    BALSAM_EXIST = False
    print("Module: 'balsam' module was not found!")


HERE = os.path.dirname(os.path.abspath(__file__))
now = "_".join(str(datetime.datetime.now(datetime.timezone.utc)).split(":")[0].split(" "))


def get_workload(wf_name):
    qs = BalsamJob.objects.filter(workflow=wf_name)
    time_data = process_job_times(qs)
    times, num_running = utilization_report(time_data)
    times = [str(t) for t in times]
    num_running = [int(n) for n in num_running]
    return times, num_running


def parseline_json(line, data):
    line = "".join(line)
    date = line.split("|")[0]
    jsn_str = line.split(">>>")[-1]
    info = json.loads(jsn_str)
    if data.get(info["type"]) == None:
        data[info["type"]] = list()
    value = info["type"]
    info["timestamp"] = date[:10] + " " + date[10:]
    info.pop("type")
    data[value].append(info)


def parseline_reward(line, data):
    data["raw_rewards"].append(float(line[-1]))


def parseline_arch_seq(line, data):
    if "'arch_seq':" in line:
        i_sta = line.index("'arch_seq':") + 1
        i_end = i_sta
        while not "]" in line[i_end]:
            i_end += 1
        l = []
        if i_sta < i_end:
            for i in range(i_sta, i_end + 1):
                value = (
                    line[i]
                    .replace("[", "")
                    .replace(",", "")
                    .replace("]", "")
                    .replace("}", "")
                )
                value = float(value) if "." in value else int(value)
                l.append(value)
        data["arch_seq"].append(l)


def parseline(line: list, data: dict, kw: str, key_: str, convert: callable):

    regexp = r"[-+]?\d*\.\d+|\d+"  # parse ints and floats

    if kw in line:
        i = line.index(kw) + 1
        info = convert(re.findall(regexp, line[i])[0])
        data[key_].append(info)


def parsing(f, data):

    # to collect hyperparameters
    kws = ["'batch_size':", "'learning_rate':", "'ranks_per_node':"]
    keys = ["batch_size", "learning_rate", "ranks_per_node"]
    converts = [int, float, int]

    line = f.readline()

    while line:
        line = line.split()
        if "y:" in line:
            parseline_reward(line, data)
            parseline_arch_seq(line, data)

            # parse algorithm hyperparameters
            for kw, key_, convert in zip(kws, keys, converts):
                parseline(line, data, kw, key_, convert)

            line = " ".join(line)
            date = line.split("|")[0]
            data["timestamps"].append(date)
        elif ">>>" in line:
            parseline_json(line, data)

        line = f.readline()


def add_subparser(subparsers):
    subparser_name = "parse"
    function_to_call = main

    parser_parse = subparsers.add_parser(
        subparser_name, help='Tool to parse "deephyper.log" and produce a JSON file.'
    )
    parser_parse.add_argument(
        "path",
        type=str,
        help=f"The parsing script takes only 1 argument: the relative path to the log file. If you want to compute the workload data with 'balsam' you should specify a path starting at least from the workload parent directory, eg. 'nas_exp1/nas_exp1_ue28s2k0/deephyper.log' where 'nas_exp1' is the workload.",
    )

    return subparser_name, function_to_call


def main(path, *args, **kwargs):
    print(f"Path to deephyper.log file: {path}")

    data = dict()
    if len(path.split("/")) >= 3:
        data["fig"] = path.split("/")[-3] + "_" + now
        workload_in_path = True
    else:
        workload_in_path = False
        data["fig"] = "data_" + now

    # initialize lists to add parsed data
    keys = [
        "raw_rewards",
        "arch_seq",
        "timestamps",
        "batch_size",
        "learning_rate",
        "ranks_per_node",
    ]
    for k in keys:
        data[k] = []

    with open(path, "r") as flog:
        print("File has been opened")
        parsing(flog, data)
    print("File closed")

    if BALSAM_EXIST and workload_in_path:
        try:
            print("Computing workload!")
            times, num_running = get_workload(path.split("/")[-3])
            data["workload"] = dict(times=times, num_running=num_running)
        except Exception as e:
            print("Exception: ", e)
            print("Failed to compute workload!...")
        else:
            print("Workload has been computed successfuly!")

    with open(data["fig"] + ".json", "w") as fjson:
        print(f'Create json file: {data["fig"]+".json"}')
        json.dump(data, fjson, indent=2)
    print("Json dumped!")

    print(f'{len(data["raw_rewards"])} evaluations collected, parsing done!')
