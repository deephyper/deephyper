"""Scikit-Optimize subpackage.

Fork of the official package: https://github.com/scikit-optimize/scikit-optimize
with extended modifications.

Scikit-Optimize, or `skopt`, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. `skopt` is reusable
in many contexts and accessible.

This fork of skopt 0.9.8 has been modified for compatibility with
DeepHyper. It should not be called directly since its parameters are
set from within the :class:`deephyper.search.Search` class.
Note that numerous additional options have been added to support additional
features of DeepHyper.
Additional subdirectories have been added containing code
to support these features.

**Note that DeepHyper users should not need to directly interact with the
interfaces in this module. All options are set from the Search class.**
The one exception to this are the utility functions and additional
multiobjective documentation provided in
:mod:`deephyper.skopt.moo`, which may be of external interest.

This module is included for the purpose of:

1. optimization developers to understand DeepHyper's optimization techniques and
2. additional documentation -- because certain optimization-related features
    (such as multiobjective features in :mod:`deephyper.skopt.moo`)
    are too nuanced to fully documented in the :class:`deephyper.search.Search`
    class's docstrings, and can be better understood through their
    documentation pages here.

Also note that although DeepHyper generally deals in maximization problems,
this module is for *minimization*.
Each search class should automatically convert the problems appropriately
before referencing this module.

In general, the :class:`deephyper.skopt.optimizer.optimizer.Optimizer` class defines the basic functionality, which
is used by the functions in :mod:`deephyper.skopt.optimizer`.
From within DeepHyper :class:`deephyper.search.Search`, the
``surrogate_model`` (str type) argument selects which of these functions
is called.
Other notable modules include :mod:`deephyper.skopt.acquisition` and
:mod:`deephyper.skopt.sampler`, selected by similar str arguments in
:class:`deephyper.search.Search`.

The multiobjective functionality (defined by :class:`deephyper.search.Search`'s
``moo_scalarization_strategy`` arg) is defined in the
:mod:`deephyper.skopt.moo` module, and used by the :class:`deephyper.skopt.optimizer.optimizer.Optimizer`.
"""

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __SKOPT_SETUP__
except NameError:
    __SKOPT_SETUP__ = False


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.9.8"

if __SKOPT_SETUP__:
    import sys

    sys.stderr.write("Partial import of skopt during the build process.\n")
    # We are not importing the rest of scikit-optimize during the build
    # process, as it may not be compiled yet
else:
    import platform
    import struct
    from . import acquisition
    from . import benchmarks
    from . import callbacks
    from . import learning
    from . import optimizer

    from . import space
    from . import sampler
    from .optimizer import dummy_minimize
    from .optimizer import forest_minimize
    from .optimizer import gbrt_minimize
    from .optimizer import gp_minimize
    from .optimizer import Optimizer
    from .searchcv import BayesSearchCV
    from .space import Space
    from .utils import dump
    from .utils import expected_minimum
    from .utils import load

    __all__ = (
        "acquisition",
        "benchmarks",
        "callbacks",
        "learning",
        "optimizer",
        "plots",
        "sampler",
        "space",
        "gp_minimize",
        "dummy_minimize",
        "forest_minimize",
        "gbrt_minimize",
        "Optimizer",
        "dump",
        "load",
        "expected_minimum",
        "BayesSearchCV",
        "Space",
    )
    IS_PYPY = platform.python_implementation() == "PyPy"
    _IS_32BIT = 8 * struct.calcsize("P") == 32
