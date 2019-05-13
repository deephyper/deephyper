import tensorflow as tf
import numpy as np
import math
import traceback
from sklearn.metrics import mean_squared_error

import deephyper.search.nas.model.arch as a
import deephyper.search.nas.model.train_utils as U
from deephyper.search import util
from deephyper.search.nas.utils._logging import JsonMessage as jm
from deephyper.search.nas.model.trainer.train_valid import TrainerTrainValid

logger = util.conf_logger('deephyper.model.trainer')


class TrainerRegressorTrainValid(TrainerTrainValid):
    def __init__(self, config, model):
        super().__init__(config, model)
