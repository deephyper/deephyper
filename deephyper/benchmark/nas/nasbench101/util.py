import collections
import copy
import os

import networkx as nx

from deephyper.benchmark.nas.nasbench101.nodes import InputNode, VariableNode
from deephyper.benchmark.nas.nasbench101.ops import (Connect, Conv1X1, Conv3X3,
                                                     Identity, MaxPool3X3)
from deephyper.benchmark.nas.nasbench101.search_space import NxSearchSpace


# Initialize the NASBench object which parses the raw data into memory (this
# should only be run once as it takes up to a few minutes).
from nasbench import api

# # Use nasbench_full.tfrecord for full dataset (run download command above).
# data_file = os.path.join(HERE, "nasbench_only108.tfrecord")
# data_file = os.path.join(HERE, "nasbench_full.frecord")
# nasbench = api.NASBench(data_file)



def create_conv_node():
    vnode = VariableNode()
    ops = [Identity, Conv1X1, Conv3X3, MaxPool3X3]
    for op in ops:
        vnode.add_op(op())
    return vnode

def create_search_space(input_shape=None, output_shape=None):
    ss = NxSearchSpace()
    nodes = []
    in_node = InputNode()

    anchors = collections.deque([in_node], maxlen=2)

    vnode = create_conv_node()
    ss.connect(in_node, vnode)
    anchors.append(vnode)

    for i in range(3):
        skipco_node = VariableNode()
        for source in anchors:
            skipco_node.add_op(Connect(ss, source))

        vnode = create_conv_node()
        ss.connect(skipco_node, vnode)
        anchors.append(vnode)

    return ss

def evaluate_ops(ss, ops, nasbench):
    ss = copy.deepcopy(ss)
    ss.set_ops(ops)

    graph = ss.contract()
    matrix = nx.adjacency_matrix(graph).todense()
    ops_labels = [str(n) for n in graph.nodes()]
    cell = api.ModelSpec(
        matrix=matrix.tolist(),   # output layer
        # Operations at the vertices of the module, matches order of matrix.
        ops=ops_labels
    )

    # Querying multiple times may yield different results. Each cell is evaluated 3
    # times at each epoch budget and querying will sample one randomly.
    data = nasbench.query(cell)
    return data["validation_accuracy"]
