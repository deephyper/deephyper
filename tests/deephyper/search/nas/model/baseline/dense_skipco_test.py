def test_structure():
    from deephyper.search.nas.model.baseline.dense_skipco import create_search_space
    from random import random
    struct = create_search_space()

    ops = [random() for _ in range(struct.num_nodes)]
    struct.set_ops(ops)
    model = struct.create_model()

    from tensorflow.keras.utils import plot_model

    plot_model(model, to_file=f'test_dense_skipco.png', show_shapes=True)