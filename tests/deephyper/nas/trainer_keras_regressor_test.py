import unittest

import pytest


@pytest.mark.nas
class TrainerKerasRegressorTest(unittest.TestCase):
    def test_trainer_regressor_train_valid_with_one_input(self):
        import sys
        from random import random

        import numpy as np
        from deephyper.benchmark.nas.linearReg.problem import Problem
        from deephyper.nas.trainer import BaseTrainer
        from deephyper.search import util
        from tensorflow.keras.utils import plot_model

        config = Problem.space

        config["hyperparameters"]["num_epochs"] = 2

        # load functions
        load_data = util.load_attr_from(config["load_data"]["func"])
        config["load_data"]["func"] = load_data
        config["create_search_space"]["func"] = util.load_attr_from(
            config["create_search_space"]["func"]
        )

        # Loading data
        kwargs = config["load_data"].get("kwargs")
        (tX, ty), (vX, vy) = load_data() if kwargs is None else load_data(**kwargs)

        print("[PARAM] Data loaded")
        # Set data shape
        input_shape = np.shape(tX)[1:]  # interested in shape of data not in length
        output_shape = np.shape(ty)[1:]

        config["data"] = {"train_X": tX, "train_Y": ty, "valid_X": vX, "valid_Y": vy}

        search_space = config["create_search_space"]["func"](
            input_shape, output_shape, **config["create_search_space"]["kwargs"]
        )
        arch_seq = [random() for i in range(search_space.num_nodes)]
        print("arch_seq: ", arch_seq)
        search_space.set_ops(arch_seq)
        search_space.plot("trainer_keras_regressor_test.dot")

        if config.get("preprocessing") is not None:
            preprocessing = util.load_attr_from(config["preprocessing"]["func"])
            config["preprocessing"]["func"] = preprocessing
        else:
            config["preprocessing"] = None

        model = search_space.create_model()
        plot_model(model, to_file="trainer_keras_regressor_test.png", show_shapes=True)

        trainer = BaseTrainer(config=config, model=model)

        res = trainer.train()
        assert res != sys.float_info.max

    def test_trainer_regressor_train_valid_with_multiple_ndarray_inputs():
        import sys
        from random import random

        import numpy as np
        from deephyper.benchmark.nas.linearReg.problem import Problem
        from deephyper.nas.trainer import BaseTrainer
        from deephyper.search import util
        from tensorflow.keras.utils import plot_model
        from deephyper.benchmark.nas.linearRegMultiInputs.problem import Problem

        config = Problem.space

        config["hyperparameters"]["num_epochs"] = 2

        # load functions
        load_data = util.load_attr_from(config["load_data"]["func"])
        config["load_data"]["func"] = load_data
        config["create_search_space"]["func"] = util.load_attr_from(
            config["create_search_space"]["func"]
        )

        # Loading data
        kwargs = config["load_data"].get("kwargs")
        (tX, ty), (vX, vy) = load_data() if kwargs is None else load_data(**kwargs)

        print("[PARAM] Data loaded")
        # Set data shape
        # interested in shape of data not in length
        input_shape = [np.shape(itX)[1:] for itX in tX]
        output_shape = list(np.shape(ty))[1:]

        config["data"] = {"train_X": tX, "train_Y": ty, "valid_X": vX, "valid_Y": vy}

        search_space = config["create_search_space"]["func"](
            input_shape, output_shape, **config["create_search_space"]["kwargs"]
        )
        arch_seq = [random() for i in range(search_space.num_nodes)]
        print("arch_seq: ", arch_seq)
        search_space.set_ops(arch_seq)
        search_space.plot("trainer_keras_regressor_test.dot")

        if config.get("preprocessing") is not None:
            preprocessing = util.load_attr_from(config["preprocessing"]["func"])
            config["preprocessing"]["func"] = preprocessing
        else:
            config["preprocessing"] = None

        model = search_space.create_model()
        plot_model(model, to_file="trainer_keras_regressor_test.png", show_shapes=True)

        trainer = BaseTrainer(config=config, model=model)

        res = trainer.train()
        assert res != sys.float_info.max

    def test_trainer_regressor_train_valid_with_multiple_generator_inputs():
        import sys

        from deephyper.benchmark.nas.linearReg.problem import Problem
        from deephyper.nas.trainer import BaseTrainer
        from tensorflow.keras.utils import plot_model
        from deephyper.benchmark.nas.linearRegMultiInputsGen.problem import Problem
        from deephyper.nas.run._util import get_search_space, load_config, setup_data

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
