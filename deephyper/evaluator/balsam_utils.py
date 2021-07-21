import json
import logging
from functools import wraps

from deephyper.evaluator.encoder import Encoder as JSONEncoder

logger = logging.getLogger(__name__)


def balsamjob_spec(run_func):
    @wraps(run_func)
    def labelled_run(param_dict):
        return run_func(param_dict)

    labelled_run._balsamjob_spec = True
    return labelled_run


def to_encodable(d):
    return json.loads(json.dumps(d, cls=JSONEncoder))
