import deephyper
from deephyper.core.exceptions import DeephyperRuntimeError
import tensorflow as tf


def import_callback(cb_name: str) -> tf.keras.callbacks.Callback:
    """Import a callback class from its name.

    Args:
        cb_name (str): class name of the callback to import fron ``tensorflow.keras.callbacks`` or ``deephyper.contrib.callbacks``.

    Raises:
        DeephyperRuntimeError: raised if the class name of the callback is not registered in corresponding packages.

    Returns:
        tensorflow.keras.callbacks.Callback: the class corresponding to the given class name.
    """
    if cb_name in dir(tf.keras.callbacks):
        return getattr(tf.keras.callbacks, cb_name)
    elif cb_name in dir(deephyper.contrib.callbacks):
        return getattr(deephyper.contrib.callbacks, cb_name)
    else:
        raise DeephyperRuntimeError(
            f"Callback '{cb_name}' is not registered in tensorflow.keras and deephyper.contrib.callbacks."
        )
