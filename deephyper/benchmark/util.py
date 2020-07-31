import hashlib
import json
import pickle
from collections import namedtuple
import time
import os
import uuid
import types
import numpy as np
from numpy import integer, floating, ndarray
from datetime import datetime
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def balsamjob_spec(run_func):
    @wraps(run_func)
    def labelled_run(param_dict):
        return run_func(param_dict)

    labelled_run._balsamjob_spec = True
    return labelled_run


class JSONEncoder(json.JSONEncoder):
    """
    Enables JSON dump of numpy data
    """

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return obj.hex
        if isinstance(obj, integer):
            return int(obj)
        elif isinstance(obj, floating):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, types.FunctionType):
            return f"{obj.__module__}.{obj.__name__}"
        else:
            return super(JSONEncoder, self).default(obj)


def to_encodable(d):
    return json.loads(json.dumps(d, cls=JSONEncoder))


def str2bool(s):
    s = s.lower().strip()
    if s == "false":
        return False
    else:
        return True


class Timer:
    def __init__(self):
        self.t0 = 0.0
        self.name = None

    def start(self, name):
        self.name = name
        self.t0 = time.time()

    def end(self):
        elapsed = time.time() - self.t0
        if not self.name:
            return
        print(f"TIMER {self.name}: {elapsed:.4f} seconds")
        self.t0 = 0.0
        self.name = None


def extension_from_parameters(param_dict):
    EXCLUDE_PARAMS = [
        "epochs",
        "model_path",
        "data_source",
        "stage_in_destination",
        "version",
        "backend",
    ]
    extension = ""
    for key in sorted(param_dict):
        if key not in EXCLUDE_PARAMS:
            extension += ".{}={}".format(key, param_dict[key])
    print("extension:", extension)
    return extension


def save_meta_data(data, filename):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_meta_data(filename):
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    return data


def resume_from_disk(benchmark_name, param_dict, data_dir="", custom_objects={}):
    from keras.models import load_model

    SavedModel = namedtuple(
        "SavedModel", ["model", "model_path", "initial_epoch", "model_mda_path"]
    )
    extension = extension_from_parameters(param_dict)
    hex_name = hashlib.sha224(extension.encode("utf-8")).hexdigest()
    model_name = "{}-{}.h5".format(benchmark_name, hex_name)
    model_mda_name = "{}-{}.pkl".format(benchmark_name, hex_name)

    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    model_path = os.path.join(data_dir, model_name)
    model_mda_path = os.path.join(data_dir, model_mda_name)

    initial_epoch = 0
    model = None

    if os.path.exists(model_path) and os.path.exists(model_mda_path):
        print("model and meta data exists; loading model from h5 file")
        if benchmark_name == "regression":
            with open(model_path, "rb") as fp:
                model = pickle.load(fp)
        else:
            model = load_model(model_path, custom_objects=custom_objects)

        saved_param_dict = load_meta_data(model_mda_path)
        initial_epoch = saved_param_dict["epochs"]
        if initial_epoch < param_dict["epochs"]:
            print(f"loading from epoch {initial_epoch}")
            print(f"running to epoch {param_dict['epochs']}")
        else:
            raise RuntimeError(
                "Specified Epochs is less than the initial epoch; will not run"
            )
    else:
        print("Did not find saved model at", model_path)

    return SavedModel(
        model=model,
        model_path=model_path,
        model_mda_path=model_mda_path,
        initial_epoch=initial_epoch,
    )


def stage_in(file_names, source, dest):
    from keras.utils.data_utils import get_file

    print("Stage in files:", file_names)
    print("From source dir:", source)
    print("To destination:", dest)

    paths = {}
    for name in file_names:
        origin = os.path.join(source, name)
        assert os.path.exists(origin), f"{origin} not found"

        if os.path.exists(dest):
            target = os.path.join(dest, name)
            paths[name] = get_file(fname=target, origin="file://" + origin)
        else:
            paths[name] = origin

        print(f"File {name} will be read from {paths[name]}")
    return paths


def numpy_dict_cache(cache_loc):
    def _cache(data_loader):
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_loc):
                logger.debug("Reading data from cache")
                with open(cache_loc, "rb") as fp:
                    return {k: arr for k, arr in np.load(fp).items()}
            else:
                logger.debug("Data not cached; invoking user data loader")
                data = data_loader(*args, **kwargs)
                with open(cache_loc, "wb") as fp:
                    np.savez(fp, **data)
                return data

        return wrapper

    return _cache
