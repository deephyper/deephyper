"""Submodule from where ``MPI`` should always be imported within the package."""

import mpi4py

# The following configuration avoid initializing MPI during the import
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI  # noqa: E402, F401
from mpi4py.futures import MPICommExecutor  # noqa: E402, F401
