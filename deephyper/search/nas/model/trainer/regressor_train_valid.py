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
            min_mse = math.inf
            for i in range(num_epochs):
                self.model.fit(
                    self.dataset_train,
                    epochs=1,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks
                )

                y_orig, y_pred = self.predict()

                try:
                    unnormalize_mse = mean_squared_error(y_orig, y_pred)
                except ValueError as err:
                    logger.error(traceback.format_exc())
                    unnormalize_mse = np.finfo('float32').max
                except:
                    raise

                # self.train_history[f'{self.metrics_name[0]}_valid'] = unnormalize_mse

                min_mse = min(min_mse, unnormalize_mse)
                logger.info(jm(epoch=i, validation_mse=float(unnormalize_mse)))

            logger.info(jm(type='result', mse=float(min_mse)))
            return min_mse
        elif num_epochs == 0:
            y_orig, y_pred = self.predict()

            try:
                unnormalize_mse = mean_squared_error(y_orig, y_pred)
            except ValueError as err:
                logger.error(traceback.format_exc())
                unnormalize_mse = np.finfo('float32').max
            except:
                raise

            logger.info(jm(epoch=0, validation_mse=float(unnormalize_mse)))
            logger.info(jm(type='result', mse=float(unnormalize_mse)))
            return unnormalize_mse
        else:
            raise RuntimeError(f'Number of epochs should be >= 0: {num_epochs}')
