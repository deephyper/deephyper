def test_create_structure():
    from random import random, seed
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.search.nas.model.baseline.anl_conv_mlp_1 import create_structure
    from deephyper.core.model_utils import number_parameters
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    seed(10)

    structure = create_structure([(10,5), (8,3)], (1,), 1)
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]
    # ops = [0 for i in range(structure.num_nodes)]
    print('num ops: ', len(ops))
    structure.set_ops(ops)
    structure.draw_graphviz('graph_anl_conv_mlp_1_test.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='graph_anl_conv_mlp_1_test.png', show_shapes=True)

    import numpy as np
    x0 = np.zeros((1, 10, 5))
    x1 = np.zeros((1, 8, 3))
    inpts = [x0, x1]
    y = model.predict(inpts)

    for x in inpts:
        print(f'shape(x): {np.shape(x)}')
    print(f'shape(y): {np.shape(y)}')

    total_parameters = number_parameters()
    print('toal_parameters: ', total_parameters)

    # assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()
