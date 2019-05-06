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

    def train(self, num_epochs=None):
        num_epochs = self.num_epochs if num_epochs is None else num_epochs

        if num_epochs > 0:
            max_rmetric = -math.inf
            for i in range(num_epochs):
                self.model.fit(
                    self.dataset_train,
                    epochs=1,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks
                )

                y_orig, y_pred = self.predict()

                try:
                    unnormalize_rmetric = sess.run(self.reward_metric_op, {
                        self.y_true_ph: y_orig,
                        self.y_pred_ph: y_pred
                    })
                except ValueError as err:
                    logger.error(traceback.format_exc())
                    unnormalize_rmetric = np.finfo('float32').min
                except:
                    raise

                max_rmetric = min(max_rmetric, unnormalize_rmetric)
                logger.info(
                    jm(epoch=i, rmetric=float(unnormalize_rmetric)))

            logger.info(jm(type='result', rmetric=float(max_rmetric)))
            return max_rmetric

        elif num_epochs == 0:
            y_orig, y_pred = self.predict()

            try:
                unnormalize_rmetric = sess.run(self.reward_metric_op, {
                    self.y_true_ph: y_orig,
                    self.y_pred_ph: y_pred
                })
            except ValueError as err:
                logger.error(traceback.format_exc())
                unnormalize_rmetric = np.finfo('float32').min
            except:
                raise

            logger.info(jm(epoch=0, rmetric=float(unnormalize_rmetric)))
            logger.info(jm(type='result', rmetric=float(unnormalize_rmetric)))
            return unnormalize_rmetric
        else:
            raise RuntimeError(
                f'Number of epochs should be >= 0: {num_epochs}')
