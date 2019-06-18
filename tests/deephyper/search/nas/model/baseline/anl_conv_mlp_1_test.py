def test_create_structure():
    from random import random, seed
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.search.nas.model.baseline.anl_conv_mlp_1 import create_structure
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    seed(10)
    shapes = [(942, ), (3820, ), (3820, )]
    structure = create_structure(shapes, (1,), 5)
    assert type(structure) is KerasStructure

    # ops = [random() for i in range(structure.num_nodes)]
    # ops = [0 for i in range(structure.num_nodes)]
    ops = [0.2, 0.8, 0.0, 0.8, 0.0, 0.0, 0.6, 0.8, 0.0, 0.2, 0.6, 0.2, 0.0, 0.4, 0.8, 0.6, 0.2, 0.8, 0.4, 0.0, 0.4, 0.8,
           0.2, 0.8, 0.4, 0.0, 0.6, 0.4, 0.4, 0.0, 0.6, 0.4, 0.0, 0.8, 0.0, 0.6, 0.6, 0.4, 0.4, 0.2, 0.0, 0.6, 0.2, 0.2, 0.4]
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('graph_anl_conv_mlp_1_test.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='graph_anl_conv_mlp_1_test.png', show_shapes=True)

    import numpy as np
    x0 = np.zeros((1, *shapes[0]))
    x1 = np.zeros((1, *shapes[1]))
    x2 = np.zeros((1, *shapes[2]))
    inpts = [x0, x1, x2]
    y = model.predict(inpts)

    for x in inpts:
        print(f'shape(x): {np.shape(x)}')
    print(f'shape(y): {np.shape(y)}')

    # assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'


if __name__ == '__main__':
    test_create_structure()
