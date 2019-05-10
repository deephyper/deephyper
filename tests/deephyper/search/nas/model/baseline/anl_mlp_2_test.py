def test_create_structure():
    from random import random, seed
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.search.nas.model.baseline.anl_mlp_2 import create_structure
    from deephyper.core.model_utils import number_parameters
    seed(10)

    structure = create_structure([(10,1), (10,1)], (1,), 5)
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]
    structure.set_ops(ops)
    structure.draw_graphviz('graph_anl_mlp_2_test.dot')

    model = structure.create_model()

    import numpy as np
    x = np.zeros((1, 10, 1))
    y = model.predict([x, x])

    print(f'shape(x): {np.shape(x)}')
    print(f'shape(y): {np.shape(y)}')

    nparameters = number_parameters()
    print('number of parameters: ', nparameters)

    assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()