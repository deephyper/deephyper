import tensorflow as tf
import numpy as np

from deephyper.search.nas.model.trainer.train_valid import TrainerTrainValid
import deephyper.search.nas.model.arch as a
import deephyper.search.nas.model.train_utils as U
from deephyper.search import util
from deephyper.search.nas.utils._logging import JsonMessage as jm

logger = util.conf_logger('deephyper.model.trainer')


class TrainerClassifierTrainValid(TrainerTrainValid):
    def __init__(self, config, model):
        super().__init__(config, model)
