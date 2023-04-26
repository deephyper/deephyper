"""DeepHyper's stopper integration module with common machine learning libraries."""
__all__ = ["DeepXDEStopperCallback"]
from deephyper.stopper.integration._deepxde_callback import DeepXDEStopperCallback


try:
    from deephyper.stopper.integration._tf_keras_callback import (  # noqa: F401
        TFKerasStopperCallback,
    )

    __all__.append("TFKerasStopperCallback")

except ImportError:
    pass
