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

    def train(self, num_epochs=None):
        num_epochs = self.num_epochs if num_epochs is None else num_epochs

        if num_epochs > 0:
            max_acc = 0
            for i in range(num_epochs):
                self.model.fit(
                    self.dataset_train,
                    epochs=1,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks
                )

                valid_info = self.model.evaluate(self.dataset_valid, steps=self.valid_steps_per_epoch)

                valid_loss, valid_acc = valid_info[0], valid_info[1]*100

                max_acc = max(max_acc, valid_acc)
                logger.info(jm(epoch=i, validation_loss=valid_loss, validation_acc=float(valid_acc)))
            logger.info(jm(type='result', acc=float(max_acc)))
            return max_acc
        elif num_epochs == 0:
            valid_info = self.model.evaluate(self.dataset_valid, steps=self.valid_steps_per_epoch)

            valid_loss, valid_acc = valid_info[0], valid_info[1]*100

            logger.info(jm(epoch=0, validation_loss=valid_loss, validation_acc=float(valid_acc)))
            logger.info(jm(type='result', acc=float(valid_acc)))
            return valid_acc
        else:
            raise RuntimeError(f'Number of epochs should be >= 0: {num_epochs}')
