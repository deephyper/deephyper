"""Utility for JobLib.

Necessary to pass the global configuration to the joblib workers until
https://github.com/joblib/joblib/issues/1071 is fixed.
"""

import sklearn
from packaging.version import parse as parse_version

# if sklearn >= 1.2.2 then use Parallel/delayed from there
# otherwise use from joblib
if isinstance(sklearn.__version__, str) and parse_version(sklearn.__version__) >= parse_version(
    "1.2.2"
):
    from sklearn.utils.parallel import Parallel, delayed  # noqa: F401
else:
    from joblib import Parallel, delayed  # noqa: F401
