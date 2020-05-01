import os
from deephyper.benchmark.nas.nasbench101.util import create_search_space, evaluate_ops
from nasbench import api

HERE = os.path.dirname(os.path.abspath(__file__))

# # Use nasbench_full.tfrecord for full dataset (run download command above).
data_file = os.path.join(HERE, "nasbench_full.tfrecord")
nasbench = api.NASBench(data_file)

def run(config):

    ss = create_search_space()

    ops = config["arch_seq"]
    val_acc = evaluate_ops(ss, ops, nasbench)

    return val_acc