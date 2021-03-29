import json
import os
from datetime import datetime

import numpy as np
from deephyper.core.exceptions.problem import WrongProblemObjective
from deephyper.core.utils import create_dir
from deephyper.search import util
from deephyper.evaluator.evaluate import Encoder

logger = util.conf_logger("deephyper.search.nas.run")


default_callbacks_config = {
    "EarlyStopping": dict(
        monitor="val_loss", min_delta=0, mode="min", verbose=0, patience=0
    ),
    "ModelCheckpoint": dict(
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
        filepath="model.h5",
        save_weights_only=False,
    ),
    "TensorBoard": dict(
        log_dir="",
        histogram_freq=0,
        batch_size=32,
        write_graph=False,
        write_grads=False,
        write_images=False,
        update_freq="epoch",
    ),
    "CSVLogger": dict(filename="training.csv", append=True),
    "CSVExtendedLogger": dict(filename="training.csv", append=True),
    "TimeStopping": dict(),
    "ReduceLROnPlateau":dict(monitor="val_loss", mode="auto", verbose=0, patience=5)
}


def load_config(config):
    # ! load functions
    config["load_data"]["func"] = util.load_attr_from(config["load_data"]["func"])

    # load augmentation strategy
    if not config.get("augment") is None:
        config["augment"]["func"] = util.load_attr_from(config["augment"]["func"])

    # load the function creating the search space
    config["create_search_space"]["func"] = util.load_attr_from(
        config["create_search_space"]["func"]
    )

    if not config.get("preprocessing") is None:
        config["preprocessing"]["func"] = util.load_attr_from(
            config["preprocessing"]["func"]
        )
    else:
        config["preprocessing"] = None

    if type(config["objective"]) is str and "." in config["objective"]:
        config["objective"] = util.load_attr_from(config["objective"])


def setup_data(config, add_to_config=True):
    # Loading data
    load_data = config["load_data"]["func"]
    kwargs = config["load_data"].get("kwargs")
    data = load_data() if kwargs is None else load_data(**kwargs)
    logger.info(f"Data loaded with kwargs: {kwargs}")

    # Set data shape
    if type(data) is tuple:
        if len(data) != 2:
            raise RuntimeError(
                f"Loaded data are tuple, should ((training_input, training_output), (validation_input, validation_output)) but length=={len(data)}"
            )
        (t_X, t_y), (v_X, v_y) = data
        if (
            type(t_X) is np.ndarray
            and type(t_y) is np.ndarray
            and type(v_X) is np.ndarray
            and type(v_y) is np.ndarray
        ):
            input_shape = np.shape(t_X)[1:]
            output_shape = np.shape(t_y)[1:]
        elif (
            type(t_X) is list
            and type(t_y) is np.ndarray
            and type(v_X) is list
            and type(v_y) is np.ndarray
        ):
            # interested in shape of data not in length
            input_shape = [np.shape(itX)[1:] for itX in t_X]
            output_shape = np.shape(t_y)[1:]
        elif (
            type(t_X) is np.ndarray
            and type(t_y) is list
            and type(v_X) is np.ndarray
            and type(v_y) is list
        ):
            # interested in shape of data not in length
            input_shape = np.shape(t_X)[1:]
            output_shape = [np.shape(ity)[1:] for ity in t_y]
        elif (
            type(t_X) is list
            and type(t_y) is list
            and type(v_X) is list
            and type(v_y) is list
        ):
            # interested in shape of data not in length
            input_shape = [np.shape(itX)[1:] for itX in t_X]
            output_shape = [np.shape(ity)[1:] for ity in t_y]
        else:
            raise RuntimeError(
                f"Data returned by load_data function are of a wrong type: type(t_X)=={type(t_X)},  type(t_y)=={type(t_y)}, type(v_X)=={type(v_X)}, type(v_y)=={type(v_y)}"
            )
        if add_to_config:
            config["data"] = {
                "train_X": t_X,
                "train_Y": t_y,
                "valid_X": v_X,
                "valid_Y": v_y,
            }
    elif type(data) is dict:
        if add_to_config:
            config["data"] = data
        input_shape = [
            data["shapes"][0][f"input_{i}"] for i in range(len(data["shapes"][0]))
        ]
        output_shape = data["shapes"][1]
    else:
        raise RuntimeError(
            f"Data returned by load_data function are of an unsupported type: {type(data)}"
        )

    if (
        output_shape == ()
    ):  # basicaly means data with shape=(num_elements) == (num_elements, 1)
        output_shape = (1,)

    logger.info(f"input_shape: {input_shape}")
    logger.info(f"output_shape: {output_shape}")

    return input_shape, output_shape


def get_search_space(config, input_shape, output_shape, seed):
    create_search_space = config["create_search_space"]["func"]
    cs_kwargs = config["create_search_space"].get("kwargs")
    if cs_kwargs is None:
        search_space = create_search_space(input_shape, output_shape, seed=seed)
    else:
        search_space = create_search_space(
            input_shape, output_shape, seed=seed, **cs_kwargs
        )
    return search_space


def setup_search_space(config, input_shape, output_shape, seed):

    search_space = get_search_space(config, input_shape, output_shape, seed)

    arch_seq = config["arch_seq"]
    logger.info(f"actions list: {arch_seq}")
    search_space.set_ops(arch_seq)

    return search_space


def compute_objective(objective, history):
    # set a multiplier to turn objective to its negative
    if type(objective) is str:
        if objective[0] == "-":
            multiplier = -1
            objective = objective[1:]
        else:
            multiplier = 1

    if type(objective) is str and ("__" in objective or objective in history):
        split_objective = objective.split("__")
        kind = split_objective[1] if len(split_objective) > 1 else "last"
        mname = split_objective[0]
        if kind == "min":
            res = min(history[mname])
        elif kind == "max":
            res = max(history[mname])
        else:  # 'last' or else, by default it will be the last one
            res = history[mname][-1]
        return multiplier * res
    elif callable(objective):
        func = objective
        return func(history)
    else:
        raise WrongProblemObjective(objective)


def preproc_trainer(config):

    if type(config["objective"]) is str:
        last_only = "__last" in config["objective"]
    else:  # should be callable
        last_only = "__last" in config["objective"].__name__

    with_pred = (
        not type(config["objective"]) is str
        and "with_pred" in config["objective"].__name__
    )
    return last_only, with_pred


def hash_arch_seq(arch_seq: list) -> str:
    return "_".join([str(el) for el in arch_seq])


def save_history(log_dir: str, history: dict, config: dict):
    if not (log_dir is None):
        history_path = os.path.join(log_dir, "history")
        if not (os.path.exists(history_path)):
            create_dir(history_path)
        now = datetime.now()
        now = now.strftime("%d-%b-%Y_%H-%M-%S")
        history_path = os.path.join(
            history_path, f"{now}oo{hash_arch_seq(config['arch_seq'])}.json"
        )
        logger.info(f"Saving history at: {history_path}")

        # convert numpy types to json compatible types
        # print(history)
        # for k,v in history.items():
        #     if type(v) is np.ndarray or:
        #         if v.dtype == np.int32 or v.dtype == np.int64:
        #             history[k] = v.astype(int).tolist()
        #         elif v.dtype == np.float32 or v.dtype == np.float64:
        #             history[k] = v.astype(float).tolist()
        #     elif type(v) is np.float32 or type(v) is np.float64:
        #         history[k] = float(v)
        #     elif type(v) is np.int32 or type(v) is np.int64:
        #         history[k] = int(v)

        with open(history_path, "w") as f:
            json.dump(history, f, cls=Encoder)