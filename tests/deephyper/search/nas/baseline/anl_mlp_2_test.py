def test_create_structure():
    # from random import random
    from deephyper.search.nas.model.space.structure import KerasStructure
    from deephyper.search.nas.model.baseline.anl_mlp_2 import create_structure
    structure = create_structure((10,), (1,), 10)
    assert type(structure) is KerasStructure
    # ops = [random() for i in range(structure.num_nodes)]
    # structure.set_ops(ops)
    # structure.draw_graphviz('graph.dot')
