def test_structure():
    from deephyper.search.nas.model.baseline.simple import create_structure
    from random import random
    struct = create_structure()

    ops = [random() for _ in range(struct.num_nodes)]
    struct.set_ops(ops)
    model = struct.create_model()

    from tensorflow.keras.utils import plot_model

    plot_model(model, to_file=f'test_structure.png', show_shapes=True)