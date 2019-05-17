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

Example:
::

    >>> import tensorflow as tf
    >>> from deephyper.search.nas.model.space.node import VariableNode
    >>> vnode = VariableNode("VNode1")
    >>> from deephyper.search.nas.model.space.op.op1d import Dense
    >>> vnode.add_op(Dense(
    ... units=10,
    ... activation=tf.nn.relu))
    >>> vnode.num_ops
    1
    >>> vnode.add_op(Dense(
    ... units=1000,
    ... activation=tf.nn.tanh))
    >>> vnode.num_ops
    2
    >>> vnode.set_op(0)
    >>> vnode.op.units
    100
    >>> str(vnode)
    'VNode1(1)(Variable[Dense_100_relu])'





ConstantNode
------------

.. autoclass:: deephyper.search.nas.model.space.node.ConstantNode
    :members:


Example:
::

    >>> import tensorflow as tf
    >>> from deephyper.search.nas.model.space.node import ConstantNode
    >>> from deephyper.search.nas.model.space.op.op1d import Dense
    >>> cnode = ConstantNode(op=Dense(units=100, activation=tf.nn.relu), name='CNode1')
    >>> str(cnode)
    'CNode1(2)(Constant[Dense_100_relu])'



MirrorNode
----------

.. autoclass:: deephyper.search.nas.model.space.node.MirrorNode
    :members:

Block
=====

.. autoclass:: deephyper.search.nas.model.space.block.Block
    :members:

Cell
====

.. autoclass:: deephyper.search.nas.model.space.cell.Cell
    :members:

Structure
=========

.. autoclass:: deephyper.search.nas.model.space.structure.KerasStructure
    :members:

.. _what-is-structure:

What is a Structure ?
=====================

.. WARNING::
    If you want to output the dot files of graphs that you are creating with the nas api please install pygraphviz: ``pip install pygrapviz``

In neural architecture search the user needs to define a structure of architecture to run the search. This structure describes the search space of the search.

Definition
----------

Formally a structure :math:`S \in \mathcal{S}` can be described as a triplet of :math:`I_S` the input space, :math:`(C_0, ..., C_{K-1})` a tuple of :math:`K \in \mathbb{N}` cells and :math:`R_{S_{out}}` a rule to apply to create the output of :math:`S`:

:math:`S = (I_S, (C_0, ..., C_{K-1}), R_{S_{out}})`

A cell :math:`C \in \mathcal{C}` is a pair of a tuple of :math:`L \in \mathbb{N}` blocks and :math:`R_{C_{out}}` a rule to apply to create the output of the current cell:

:math:`C = ( (B_0, ..., B_{L-1}), R_{C_{out}})`

A block :math:`B \in \mathcal{B}` is a tuple of :math:`M \in \mathbb{N}` nodes:

:math:`B = ( N_0, ..., N_{M-1} )`

A node :math:`N \in \mathcal{N}` is a pair of an ordered set of :math:`N \in \mathbb{N}` possible operations :math:`\{O_0, ..., O_{N-1}\}` and a choosen operation :math:`O_i \in \{O_0, ..., O_{N-1}\}`:

:math:`N = ( O_i, \{O_0, ..., O_{N-1}\} )`

Visual example
--------------

The following figure is an example of structure which is defined in order to search over fully connected networks. This structure is defined with 2 cells. Each cell contains 1 block. Each block contains 3 nodes. The first node will create a :math:`Connect` operation in order to choose the input of the current block. The second node will choose an operation of kind :math:`Dense(x, y)` where :math:`x` is the number of units in a :math:`Dense` layer and :math:`y` is a string which represents an activation function. A :math:`Dense` layer means that we are doing this operation: :math:`f_{activation}(WX + B)` where :math:`W \in \mathcal{M}_{m,n}(\mathbb{R})` is a matrix of parameters called weights, :math:`X \in \mathbb{R}^n` is a vector of inputs and :math:`B \in \mathbb{R}^m` is a vector of parameters called bias.


.. figure:: ex1_dense_structure.png
   :scale: 55 %
   :alt: example_dense_structure
   :align: center

Code example 1
--------------

The following code shows functions in python to create the previous example structure. This functions are located in ``deephyper.search.nas.model.baseline.anl_mlp_1``:
::

    import tensorflow as tf

    from deephyper.search.nas.model.baseline.util.struct import create_struct_full_skipco
    from deephyper.search.nas.model.space.block import Block
    from deephyper.search.nas.model.space.cell import Cell
    from deephyper.search.nas.model.space.node import VariableNode
    from deephyper.search.nas.model.space.op.basic import Connect
    from deephyper.search.nas.model.space.op.op1d import (Dense, Identity,
                                                      Dropout)


    def create_dense_cell_type1(input_nodes):
        """Dense type 1

        Args:
            input_nodes (list(VariableNode)): possible inputs of the current cell.

        Returns:
            Cell: a Cell instance.
        """
        cell = Cell(input_nodes)

        def create_block():
            # creation of node N_0
            n0 = VariableNode('N_0')
            for inpt in input_nodes:
                n0.add_op(Connect(cell.graph, inpt, n0))

            # list of operations for node N_1
            mlp_op_list = [
                Identity(),
                Dense(5, tf.nn.relu),
                Dense(5, tf.nn.tanh),
                Dense(10, tf.nn.relu),
                Dense(10, tf.nn.tanh),
                Dense(20, tf.nn.relu),
                Dense(20, tf.nn.tanh)
            ]
            # creation of node N_1
            n1 = VariableNode('N_1')
            for op in mlp_op_list:
                n1add_op(op)

            # list of operations for node N_2
            dropout_ops = [
                Dropout(0.),
                Dropout(0.1),
                Dropout(0.2),
                Dropout(0.3),
                Dropout(0.4),
                Dropout(0.5),
                Dropout(0.6)
            ]
            # creation of node N_2
            n2 = VariableNode('N_2')
            for op in dropout_ops:
                n2.add_op(op)

            # creation of current block with nodes and
            # connections between nodes
            block = Block()
            block.add_node(n0)
            block.add_node(n1)
            block.add_node(n2)

            block.add_edge(n0, n1)
            block.add_edge(n1, n2)
            return block

        # creation of block 0
        block0 = create_block()

        # creation of block 1
        block1 = create_block()

        cell.add_block(block0)
        cell.add_block(block0)

        cell.set_outputs()
        return cell

    def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
        return create_struct_full_skipco(
            input_shape,
            output_shape,
            create_dense_cell_type1,
            num_cells)

And here is a test function which choose randomly a set of operations for the structure and then do a prediction in order to check the good construction of the tensor graph:
::

    def test_create_structure():
        from random import random
        from deephyper.search.nas.model.space.structure import KerasStructure
        from deephyper.search.nas.model.baseline.anl_mlp_1 import create_structure

        structure = create_structure((10,), (1,), 2)
        assert type(structure) is KerasStructure

        ops = [random() for i in range(structure.num_nodes)]
        structure.set_ops(ops)
        structure.draw_graphviz('graph_anl_mlp_1_test.dot')

        model = structure.create_model()

        import numpy as np
        x = np.zeros((1, 10))
        y = model.predict(x)

        assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'


Code example 2
--------------

Here is an other example of structure:

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
