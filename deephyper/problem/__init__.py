__doc__ = (
    """The ``deephyper.problem`` module is deprecated, use ``deephyper.hpo`` instead."""
)
from deephyper.core.warnings import deprecated_api

from deephyper.hpo import HpProblem

deprecated_api(__doc__)
