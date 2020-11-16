"""
Operations corresponding to convolution neural networks.

Learn more about different kind of convolution : https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
"""

import tensorflow as tf

from ..op import Operation


class Conv2D(Operation):
    """Classic convolution with 2 dimensions.

    Create a 2 dimensions classic convolution operation.
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/Conv2D

    Args:
        filter_height (int): height of a filter or kernel.
        filter_width (int): width of a filter or kernel.
        num_filters (int): number of filters in the convolution operation.
    """

    def __init__(
        self,
        kernel_size,
        filters=8,
        strides=1,
        padding="SAME",
        dilation_rate=1,
        activation=None,
    ):
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self._layer = None

    def __str__(self):
        return f"Conv2D_{self.kernel_size}_f{self.filters}"

    def __call__(self, inputs, **kwargs):
        """Create the tensorflow operation.

        Args:
            inputs (list(Tensor)): list of input tensors.

        Return: a tensor corresponding to the operation.
        """
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:
            self._layer = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                activation=self.activation,
            )
        out = self._layer(inputs[0])
        return out


class SeparableConv2D(Operation):
    """Depthwise-separable convolution with 2 dimensions.

    Create a 2 dimensions depthwise-separable convolution operation.
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/SeparableConv2D

    Args:
        filter_height (int): height of a filter or kernel.
        filter_width (int): width of a filter or kernel.
        num_filters (int): number of filters in the convolution operation.
    """

    def __init__(self, kernel_size, filters=8, strides=1, padding="same"):
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self._layer = None

    def __str__(self):
        return f"DepSepCNN2D_{self.kernel_size}_f{self.filters}"

    def __call__(self, inputs, **kwargs):
        """Create the tensorflow operation.

        Args:
            inputs (list(Tensor)): list of input tensors.

        Return: a tensor corresponding to the operation.
        """
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:
            self._layer = tf.keras.layers.SeparableConv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
            )
        out = self._layer(inputs[0])
        return out


class MaxPool2D(Operation):
    """Max pooling with 2 dimensions.

    Create a 2 dimensions max pooling operation.
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/MaxPool2D

    Args:
        kernel_height (int):
        kernel_width (int):
        stride_height (int):
        stride_width (int):
        padding (string): 'SAME' or 'VALID'
        num_filters (int): corresponding to the number of filters we need the output to have
    """

    def __init__(self, pool_size, strides=1, padding="same", num_filters=32):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.num_filters = num_filters
        self._layer = None

    def __str__(self):
        return f"MaxPool2D_k{self.pool_size}_s{self.strides}"

    def __call__(self, inputs, **kwargs):
        """Create the tensorflow operation.

        Args:
            inputs (list(Tensor)): list of input tensors.

        Return: a tensor corresponding to the operation.
        """
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:
            self._layer = tf.keras.layers.MaxPool2D(
                pool_size=self.pool_size, strides=self.strides, padding=self.padding
            )
        out = self._layer(inputs[0])
        return out


class AvgPool2D(Operation):
    """Average pooling with 2 dimensions.

    Create a 2 dimensions average pooling operation.
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/AveragePooling2D

    Args:
        kernel_height (int):
        kernel_width (int):
        stride_height (int):
        stride_width (int):
        padding (string): 'SAME' or 'VALID'
        num_filters (int): corresponding to the number of filters we need the output to have
    """

    def __init__(self, pool_size, strides=1, padding="same", num_filters=32):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.num_filters = num_filters
        self._layer = None

    def __str__(self):
        return f"AvgPool2D_k{self.pool_size}_s{self.strides}"

    def __call__(self, inputs, **kwargs):
        """Create the tensorflow operation.

        Args:
            inputs (list(Tensor)): list of input tensors.

        Return: a tensor corresponding to the operation.
        """
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:
            self._layer = tf.keras.layers.AvgPool2D(
                pool_size=self.pool_size, strides=self.strides, padding=self.padding
            )
        out = self._layer(inputs[0])
        return out
