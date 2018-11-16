"""
Neural Architecture Search models are following a generic workflow. This is why you don't need to define the function which runs the model yourself. Here is the generic workflow:
    - load data
    - preprocess data
    - build tensor graph of model (using Structure interface)
    - train model
    - return accuracy or mse

The basic generic function which is used in our package to run a model for NAS is

.. autofunction:: deephyper.search.nas.model.run.alpha.run

All the items are linked by a problem definition.
"""
