import collections

import tensorflow as tf
from deephyper.contrib.layers.evonorm import EvoNormS01D

from ..space import AutoKSearchSpace
from ..space.node import ConstantNode, MimeNode, VariableNode
from ..space.op.basic import Tensor
from ..space.op.connect import Connect
from ..space.op.merge import AddByProjecting
from ..space.op.op1d import Dense, Identity


def swish(x):
    return x * tf.nn.sigmoid(x)


def add_dense_to_(node):
    node.add_op(Identity())  # we do not want to create a layer in this case

    activations = [None, swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))


def create_search_space(
    input_shape=(10,),
    output_shape=(7,),
    num_layers=10,
    regression=True,
    bn=False,
    *args,
    **kwargs
):

    arch = AutoKSearchSpace(input_shape, output_shape, regression=regression)
    source = prev_input = arch.input_nodes[0]

    # look over skip connections within a range of the 3 previous nodes
    anchor_points = collections.deque([source], maxlen=3)

    cbn_init = False

    for _ in range(num_layers):
        vnode = VariableNode()
        add_dense_to_(vnode)

        arch.connect(prev_input, vnode)

        # * Cell output
        cell_output = vnode

        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation="relu"))

        for anchor in anchor_points:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        if bn:

            if not cbn_init:
                cbn = _cbn = VariableNode()
                cbn_init = True
            else:
                cbn = MimeNode(_cbn)

            cbn.add_op(Identity())
            cbn.add_op(tf.keras.layers.BatchNormalization())
            cbn.add_op(EvoNormS01D())
            arch.connect(cmerge, cbn)

            prev_input = cbn
        else:
            prev_input = cmerge

        # ! for next iter
        anchor_points.append(prev_input)

    return arch
