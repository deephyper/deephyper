"""Neural architecture search spaces for tabular data."""
from .dense_skipco import DenseSkipCoSpace
from .one_layer import OneLayerSpace
from .feed_forward import FeedForwardSpace
from .supervised_reg_auto_encoder import SupervisedRegAutoEncoderSpace

__all__ = [
    "DenseSkipCoSpace",
    "OneLayerSpace",
    "FeedForwardSpace",
    "SupervisedRegAutoEncoderSpace",
]
