Neural Architecture Search
**************************

Build custom cells
==================

.. autoclass:: nas.cell.Node
    :members:

.. autoclass:: nas.cell.Block
    :members:

.. autoclass:: nas.cell.Cell
    :members:

Build custom structure
======================

.. autoclass:: nas.cell.SequentialStructure
    :members:

Build in operations
===================

Basic operations
----------------

.. autoclass:: nas.operation.basic.Connect
    :members:

MLP operations
--------------

CNN operations
--------------

.. autoclass:: nas.operation.cnn.Convolution2D
    :members:

.. autoclass:: nas.operation.cnn.MaxPooling2D
    :members:

Build in cells
==============

Multi Layer Perceptron
----------------------

.. autofunction:: nas.cell.mlp.create_dense_cell_example

.. autofunction:: nas.cell.mlp.create_dense_cell_type1

Convolution
-----------

.. autofunction:: nas.cell.cnn.create_cnn_cell_type1
