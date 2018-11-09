Neural Architecture Search (NAS)
********************************

.. WARNING::
    If you want to output the dot files of graphs that you are creating with the nas api please install pygraphviz: ``pip install pygrapviz``

In neural architecture search we have an agent who is producing a sequence of actions to build an architecture. In deephyper we are bringing a specific API to define the action space of this agent which is located in two modules: ``deephyper.search.nas.cell`` and ``deephyper.search.nas.operation``. Let's start with an example of a structure with the following figure:

.. image:: ../_static/img/anl_mlp_1.png
    :scale: 50 %
    :alt: anl_mlp_1 structure with 3 cells
    :align: center

The previous image represents a specific choice of operations in a ``SequentialStructure`` with 3 cells (orange), each cell contains 5 blocks (blue), and each block contains 5 nodes. Let's take a more simple structure to understand its purpose:

.. image:: ../_static/img/anl_mlp_toy_set.png
    :align: center

In the previous figure we have a choice of operations for a very simple structure called ``anl_mlp_toy`` which can be represented with one cell by the following figure:

.. image:: ../_static/img/anl_mlp_toy_unset.png
    :align: center

In this structure we have only 1 cell which contains only 1 block. This block contains 2 nodes, the first one represents the creation of a connection (N1_2), the second one represent the creation of a multi layer Perceptron (N2_3).

Build custom cells
==================

.. autoclass:: deephyper.search.nas.cell.Node
    :members:

.. autoclass:: deephyper.search.nas.cell.Block
    :members:

.. autoclass:: deephyper.search.nas.cell.Cell
    :members:

Build custom structures
=======================

.. autoclass:: deephyper.search.nas.cell.SequentialStructure
    :members:

Build in structures
===================

.. autofunction:: deephyper.search.nas.contrib.anl_mlp_1.create_structure

.. autofunction:: deephyper.search.nas.contrib.google_nas_net.create_structure

Build in operations
===================

Basic operations
----------------

.. autoclass:: deephyper.search.nas.operation.basic.Connect
    :members:

MLP operations
--------------

.. automodule:: deephyper.search.nas.operation.mlp
    :members:

CNN operations
--------------

.. autoclass:: deephyper.search.nas.operation.cnn.IdentityConv2D
    :members:

.. autoclass:: deephyper.search.nas.operation.cnn.Convolution2D
    :members:

.. autoclass:: deephyper.search.nas.operation.cnn.DepthwiseSeparable2D
    :members:

.. autoclass:: deephyper.search.nas.operation.cnn.Dilation2D
    :members:

.. autoclass:: deephyper.search.nas.operation.cnn.MaxPooling2D
    :members:

.. autoclass:: deephyper.search.nas.operation.cnn.AvgPooling2D
    :members:

Build in cells
==============

Multi Layer Perceptron
----------------------

.. autofunction:: deephyper.search.nas.cell.mlp.create_dense_cell_type1

.. autofunction:: deephyper.search.nas.cell.mlp.create_dense_cell_type2

Convolution
-----------

.. autofunction:: deephyper.search.nas.cell.cnn.create_cnn_base_cell

.. autofunction:: deephyper.search.nas.cell.cnn.create_cnn_normal_cell

.. autofunction:: deephyper.search.nas.cell.cnn.create_cnn_reduction_cell
