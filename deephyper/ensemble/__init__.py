"""Sub-package to provides tools to build ensembles of models >

For Tensorflow/Keras 2.0 models use the ``DEEPHYPER_NN_BACKEND=tf_keras2`` environment variable and make sure to use ``tf_keras`` package. Checkpointed models can have the extension ``.h5`` or ``.keras``.

For PyTorch models use the ``DEEPHYPER_NN_BACKEND=torch`` environment variable. Checkpointed models can have the extension ``.pth``.
"""

from deephyper.ensemble._ensemble import Ensemble

from deephyper.ensemble._uq_ensemble import (
    UQEnsembleRegressor,
    UQEnsembleClassifier,
)

__all__ = [
    "Ensemble",
    "UQEnsembleRegressor",
    "UQEnsembleClassifier",
]
