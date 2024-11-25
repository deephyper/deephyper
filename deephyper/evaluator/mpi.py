import mpi4py

# The following configuration avoid initializing MPI during the import
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI  # noqa: E402, F401
