from typing import Type

import deephyper
import deephyper.core.exceptions
import tensorflow as tf


def import_callback(cb_name: str) -> Type[tf.keras.callbacks.Callback]:
    """Import a callback class from its name.

    Args:
        cb_name (str): class name of the callback to import fron ``tensorflow.keras.callbacks`` or ``deephyper.keras.callbacks``.

    Raises:
        DeephyperRuntimeError: raised if the class name of the callback is not registered in corresponding packages.

    Returns:
        tensorflow.keras.callbacks.Callback: the class corresponding to the given class name.
    """
    if cb_name in dir(tf.keras.callbacks):
        return getattr(tf.keras.callbacks, cb_name)
    elif cb_name in dir(deephyper.keras.callbacks):
        return getattr(deephyper.keras.callbacks, cb_name)
    else:
        raise deephyper.core.exceptions.DeephyperRuntimeError(
            f"Callback '{cb_name}' is not registered in tensorflow.keras and deephyper.keras.callbacks."
        )
