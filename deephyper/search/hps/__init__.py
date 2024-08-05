__doc__ = """The ``deephyper.search.hps`` module is deprecated, use ``deephyper.hpo`` instead."""
from deephyper.core.warnings import deprecated_api

from deephyper.hpo import CBO

try:
    from deephyper.hpo import MPIDistributedBO
except ImportError:
    pass

deprecated_api(__doc__)
