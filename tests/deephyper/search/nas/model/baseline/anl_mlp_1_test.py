def test_create_structure():
    from random import random
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.search.nas.model.baseline.anl_mlp_1 import create_structure

    structure = create_structure((10,), (1,), 2)
    assert type(structure) is KerasStructure

    ops = [random() for i in range(structure.num_nodes)]
    structure.set_ops(ops)
    # structure.draw_graphviz('graph_anl_mlp_1_test.dot')

    model = structure.create_model()

    import numpy as np
    x = np.zeros((1, 10))
    y = model.predict(x)

    assert np.shape(y) == (1, 1), f'Wrong output shape {np.shape(y)} should be {(1, 1)}'

if __name__ == '__main__':
    test_create_structure()