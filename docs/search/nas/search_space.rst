Search space
************

.. toctree::
   :maxdepth: 2

   layers/index
   op/index

Nodes
=====

.. autoclass:: deephyper.search.nas.model.space.node.Node
    :members:

VariableNode
------------

.. autoclass:: deephyper.search.nas.model.space.node.VariableNode
    :members:

ConstantNode
------------

.. autoclass:: deephyper.search.nas.model.space.node.ConstantNode
    :members:

MirrorNode
----------

.. autoclass:: deephyper.search.nas.model.space.node.MirrorNode
    :members:

Structure
=========

.. autoclass:: deephyper.search.nas.model.space.struct.DirectStructure
    :members:

.. _what-is-structure:

Code example 1
==============

Here is an example of structure:

::

    import tensorflow as tf
    from tensorflow.keras.utils import plot_model
    from random import random

    from deephyper.search.nas.model.space.block import Block
    from deephyper.search.nas.model.space.cell import Cell
    from deephyper.search.nas.model.space.node import (ConstantNode, MirrorNode,
                                                    VariableNode)
    from deephyper.search.nas.model.space.op.op1d import (Dense,
                                                        Dropout, Identity)
    from deephyper.search.nas.model.space.structure import KerasStructure

    # Data input shapes
    input_shapes = [(942, ), (3820, ), (3820, )]

    # Model output shape
    output_shape = (1, )

    # Creating the structure
    struct = KerasStructure(input_shapes, output_shape)

    # Getting the input nodes generated from inputs shapes
    input_nodes = struct.input_nodes

    # 1. Creating the first CELL
    cell0 = Cell(input_nodes)

    # 1.1. Creating the first BLOCK "cell0_block0" of CELL "cell1"
    cell0_block0 = Block()

    # 1.1. Saving the first input NODE with shape (942, ) for ease
    input_node = input_nodes[0]

    # 1.1.1. Creating the first and only NODE "cell0_block0_n0" of BLOCK "cell0_block0"
    #   "cell0_block0_n0" is a ConstantNode which means it has a fixed operation.
    cell0_block0_n0 = ConstantNode(op=Dense(100, tf.nn.relu), name="C0_B0_N0")

    # 1.1.1. Connecting input_node -> cell1_block1_n1
    cell0.graph.add_edge(input_node, cell0_block0_n0)

    # 1.1.1. Adding NODE "cell1_block1_n1" to BLOCK "cell1_block1"
    cell0_block0.add_node(cell0_block0_n0)

    # 1.1. Adding BLOCK "cell1_block1" to CELL "cell1"
    cell0.add_block(cell0_block0)

    # 1.2. Creating the second BLOCK "cell0_block1" of CELL "cell0"
    cell0_block1 = Block()

    # 1.2. Saving the second input NODE with shape (3820, ) for ease
    input_node = input_nodes[1]

    # 1.2.1. Creating the first and only NODE "cell0_block1_n0" of BLOCK "cell0_block1"
    #   "cell0_block1_n0" is a VariableNode which means it will have a set of possible
    #   operations.
    cell0_block1_n0 = VariableNode("C0_B1_N0")

    # 1.2.1. Adding operations to NODE "cell0_block1_n0"
    cell0_block1_n0.add_op(Identity())
    cell0_block1_n0.add_op(Dense(100, tf.nn.relu))
    cell0_block1_n0.add_op(Dropout(0.2))

    # 1.2.1. Connecting input_node -> cell0_block1_n0
    cell0.graph.add_edge(input_node, cell0_block1_n0)

    # 1.2.1. Adding NODE "cell0_block1_n0" to BLOCK "cell0_block1"
    cell0_block1.add_node(cell0_block1_n0)

    # 1.2. Adding BLOCK "cell2_block1" to CELL "cell2"
    cell0.add_block(cell0_block1)

    # 1.3. Creating the third BLOCK "cell0_block2" of CELL "cell0"
    cell0_block2 = Block()

    # 1.3. Saving the third input NODE with shape (3820, ) for ease
    input_node = input_nodes[2]

    # 1.3.1. Creating the first and only NODE "cell0_block2_n0" of BLOCK "cell0_block2"
    #   "cell0_block2_n0" is a MirrorNode which means is sharing the OPERATION choosen for
    #   "cell0_block1_n0" and its WEIGHTS.
    cell0_block2_n0 = MirrorNode(node=cell0_block1_n0)

    # 1.3.1. Connecting input_node -> cell0_block2_n0
    cell0.graph.add_edge(input_node, cell0_block2_n0)

    # 1.3.1. Adding NODE "cell0_block2_n0" to BLOCK "cell0_block2"
    cell0_block2.add_node(cell0_block2_n0)

    # 1.3. Adding BLOCK "cell2_block1" to CELL "cell2"
    cell0.add_block(cell0_block2)

    # 1. Adding CELL "cell0" to STRUCTURE "struct"
    struct.add_cell(cell0)

    # Choosing a random set of operations for all VariableNodes (here 1)
    ops = [random() for i in range(struct.num_nodes)]

    # Setting operations
    struct.set_ops(ops)

    # Visualize the corresponding set of choice
    struct.draw_graphviz('graph_candle_mlp_5.dot')

    # Building the keras model
    model = struct.create_model()

    # Visualize the keras model built
    plot_model(model, to_file='graph_candle_mlp_5.png', show_shapes=True)
