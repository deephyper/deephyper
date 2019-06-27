import traceback

import numpy as np

from deephyper.search import util
from deephyper.search.nas.model.run.util import (compute_objective,
                                                 load_config, preproc_trainer,
                                                 setup_data, setup_structure)
from deephyper.search.nas.model.trainer.train_valid import TrainerTrainValid

logger = util.conf_logger('deephyper.search.nas.run')


def run(config):
    load_config(config)

    input_shape, output_shape = setup_data(config)

    structure = setup_structure(config, input_shape, output_shape)

    model_created = False
    try:
        model = structure.create_model()
        model_created = True
    except:
        logger.info('Error: Model creation failed...')
        logger.info(traceback.format_exc())

    if model_created:
        trainer = TrainerTrainValid(config=config, model=model)

        last_only, with_pred = preproc_trainer(config)

        history = trainer.train(with_pred=with_pred, last_only=last_only)

        result = compute_objective(config['objective'], history)
    else:
        # penalising actions if model cannot be created
        result = -1

    return result
