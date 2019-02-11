def test_create_structure():
    from random import random
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.search.nas.model.baseline.anl_conv_mlp_1 import create_structure
    from tensorflow.keras.utils import plot_model

    structure = create_structure((100, 1), (10,), 10)
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]
    structure.set_ops(ops)
    structure.draw_graphviz('graph_anl_conv_mlp_1_test.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='graph_anl_conv_mlp_1_test.png')

    import numpy as np
    from math import sin
    a = np.linspace(0, 6.28, 100)
    x = np.array([[[sin(x_i)] for x_i in a]])
    y = model.predict(x)
    print(np.shape(y))

    print(f'shape(x): {np.shape(x)}')
    print(f'shape(y): {np.shape(y)}')

    import matplotlib.pyplot as plt
    plt.plot(a, x[0], 'r--')
    plt.plot(a, y[0, :], ':')
    plt.show()

    # assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()
