Neural Architecture Search
**************************

Build custom cells
==================

.. autoclass:: deephyper.search.nas.cell.Node
    :members:

.. autoclass:: deephyper.search.nas.cell.Block
    :members:

.. autoclass:: deephyper.search.nas.cell.Cell
    :members:

Build custom structure
======================

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

CNN operations
--------------

.. autoclass:: deephyper.search.nas.operation.cnn.IdentityConv2D

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

.. autofunction:: deephyper.search.nas.cell.mlp.create_dense_cell_example

.. autofunction:: deephyper.search.nas.cell.mlp.create_dense_cell_type1

.. autofunction:: deephyper.search.nas.cell.mlp.create_dense_cell_type2

Convolution
-----------

.. autofunction:: deephyper.search.nas.cell.cnn.create_cnn_base_cell

.. autofunction:: deephyper.search.nas.cell.cnn.create_cnn_normal_cell

.. autofunction:: deephyper.search.nas.cell.cnn.create_cnn_reduction_cell
