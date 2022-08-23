import unittest

import pytest


@pytest.mark.slow
@pytest.mark.nas
class TrainerKerasRegressorTest(unittest.TestCase):
    def test_trainer_regressor_train_valid_with_one_input(self):
        import sys
        from random import random

        import deephyper.core.utils
        import numpy as np
        from deephyper.nas.trainer import BaseTrainer
        from deephyper.test.nas.linearReg.problem import Problem
        from tensorflow.keras.utils import plot_model

        config = Problem.space

        config["hyperparameters"]["num_epochs"] = 2

        # load functions
        load_data = deephyper.core.utils.load_attr(config["load_data"]["func"])
        config["load_data"]["func"] = load_data
        config["search_space"]["class"] = deephyper.core.utils.load_attr(
            config["search_space"]["class"]
        )

        # Loading data
        kwargs = config["load_data"].get("kwargs")
        (tX, ty), (vX, vy) = load_data() if kwargs is None else load_data(**kwargs)

        print("[PARAM] Data loaded")
        # Set data shape
        input_shape = np.shape(tX)[1:]  # interested in shape of data not in length
        output_shape = np.shape(ty)[1:]

        config["data"] = {"train_X": tX, "train_Y": ty, "valid_X": vX, "valid_Y": vy}

        search_space = config["search_space"]["class"](
            input_shape, output_shape, **config["search_space"]["kwargs"]
        ).build()
        arch_seq = [random() for i in range(search_space.num_nodes)]
        print("arch_seq: ", arch_seq)
        search_space.set_ops(arch_seq)
        search_space.plot("trainer_keras_regressor_test.dot")

        if config.get("preprocessing") is not None:
            preprocessing = deephyper.core.utils.load_attr(
                config["preprocessing"]["func"]
            )
            config["preprocessing"]["func"] = preprocessing
        else:
            config["preprocessing"] = None

        model = search_space.create_model()
        plot_model(model, to_file="trainer_keras_regressor_test.png", show_shapes=True)

        trainer = BaseTrainer(config=config, model=model)

        res = trainer.train()
        assert res != sys.float_info.max

    def test_trainer_regressor_train_valid_with_multiple_ndarray_inputs(self):
        import sys
        from random import random

        import deephyper.core.utils
        import numpy as np
        from deephyper.nas.trainer import BaseTrainer
        from deephyper.test.nas.linearRegMultiInputs.problem import Problem
        from tensorflow.keras.utils import plot_model

        config = Problem.space

        config["hyperparameters"]["num_epochs"] = 2

        # load functions
        load_data = deephyper.core.utils.load_attr(config["load_data"]["func"])
        config["load_data"]["func"] = load_data
        config["search_space"]["class"] = deephyper.core.utils.load_attr(
            config["search_space"]["class"]
        )

        # Loading data
        kwargs = config["load_data"].get("kwargs")
        (tX, ty), (vX, vy) = load_data() if kwargs is None else load_data(**kwargs)

        print("[PARAM] Data loaded")
        # Set data shape
        # interested in shape of data not in length
        input_shape = [np.shape(itX)[1:] for itX in tX]
        output_shape = np.shape(ty)[1:]

        config["data"] = {"train_X": tX, "train_Y": ty, "valid_X": vX, "valid_Y": vy}

        print(f"{input_shape=}")
        print(f"{output_shape=}")

        search_space = config["search_space"]["class"](
            input_shape, output_shape, **config["search_space"]["kwargs"]
        ).build()
        arch_seq = [random() for i in range(search_space.num_nodes)]
        print("arch_seq: ", arch_seq)
        search_space.set_ops(arch_seq)
        search_space.plot("trainer_keras_regressor_test.dot")

        if config.get("preprocessing") is not None:
            preprocessing = deephyper.core.utils.load_attr(
                config["preprocessing"]["func"]
            )
            config["preprocessing"]["func"] = preprocessing
        else:
            config["preprocessing"] = None

        model = search_space.create_model()
        plot_model(model, to_file="trainer_keras_regressor_test.png", show_shapes=True)

        trainer = BaseTrainer(config=config, model=model)

        res = trainer.train()
        assert res != sys.float_info.max

    def test_trainer_regressor_train_valid_with_multiple_generator_inputs(self):
        import sys

        from deephyper.nas.run._util import get_search_space, load_config, setup_data
        from deephyper.nas.trainer import BaseTrainer
        from deephyper.test.nas.linearReg.problem import Problem
        from deephyper.test.nas.linearRegMultiInputsGen import Problem
        from tensorflow.keras.utils import plot_model

        config = Problem.space

        load_config(config)

        input_shape, output_shape = setup_data(config)

        search_space = get_search_space(config, input_shape, output_shape, 42)

        config["hyperparameters"]["num_epochs"] = 2

        model = search_space.sample()
        plot_model(model, to_file="trainer_keras_regressor_test.png", show_shapes=True)

        trainer = BaseTrainer(config=config, model=model)

        res = trainer.train()
        assert res != sys.float_info.max


if __name__ == "__main__":
    test = TrainerKerasRegressorTest()
    test.test_trainer_regressor_train_valid_with_multiple_ndarray_inputs()
