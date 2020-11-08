"""
Neural Architecture Search models are following a generic workflow. This is why you don't have to define the function which runs the model yourself if this pipeline corresponds to your needs. Here is the generic training pipeline:
    - load data
    - preprocess data
    - build tensor graph of model (using Structure interface)
    - train model
    - return reward

The basic generic function which is used in our package to run a model for NAS is

.. autofunction:: deephyper.nas.run.alpha.run

All the items are linked by a problem definition.
"""
