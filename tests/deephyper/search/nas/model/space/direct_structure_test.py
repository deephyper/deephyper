import pytest
from deephyper.core.exceptions.nas.struct import WrongOutputShape


@pytest.mark.incremental
class TestDirectStructure:
    def test_import(self):
        from deephyper.search.nas.model.space.struct import DirectStructure

    def test_create(self):
        from deephyper.search.nas.model.space.struct import DirectStructure
        DirectStructure((5, ), (1, ))

    def test_create_one_vnode(self):
        from deephyper.search.nas.model.space.struct import DirectStructure
        struct = DirectStructure((5, ), (1, ))

        from deephyper.search.nas.model.space.node import VariableNode
        vnode = VariableNode()

        struct.connect(struct.input_nodes[0], vnode)

        from deephyper.search.nas.model.space.op.op1d import Dense
        vnode.add_op(Dense(1))

        struct.set_ops([0])

        falias = 'test_direct_structure'
        struct.draw_graphviz(f'{falias}.dot')

        model = struct.create_model()
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=f'{falias}.png', show_shapes=True)

    def test_create_one_vnode_with_wrong_output_shape(self):
        from deephyper.search.nas.model.space.struct import DirectStructure
        struct = DirectStructure((5, ), (1, ))

        from deephyper.search.nas.model.space.node import VariableNode
        vnode = VariableNode()

        struct.connect(struct.input_nodes[0], vnode)

        from deephyper.search.nas.model.space.op.op1d import Dense
        vnode.add_op(Dense(10))

        struct.set_ops([0])

        with pytest.raises(WrongOutputShape):
            struct.create_model()

    def test_create_more_nodes(self):
        from deephyper.search.nas.model.space.struct import DirectStructure
        from deephyper.search.nas.model.space.node import VariableNode
        from deephyper.search.nas.model.space.op.op1d import Dense
        struct = DirectStructure((5, ), (1, ))

        vnode1 = VariableNode()
        struct.connect(struct.input_nodes[0], vnode1)

        vnode1.add_op(Dense(10))

        vnode2 = VariableNode()
        vnode2.add_op(Dense(1))

        struct.connect(vnode1, vnode2)

        struct.set_ops([0, 0])

        falias = 'test_direct_structure'
        struct.draw_graphviz(f'{falias}.dot')

        model = struct.create_model()
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=f'{falias}.png', show_shapes=True)

    def test_create_multiple_inputs_with_one_vnode(self):
        from deephyper.search.nas.model.space.struct import DirectStructure
        from deephyper.search.nas.model.space.node import VariableNode, ConstantNode
        from deephyper.search.nas.model.space.op.op1d import Dense
        from deephyper.search.nas.model.space.op.merge import Concatenate
        struct = DirectStructure([(5, ), (5, )], (1, ))

        merge = ConstantNode()
        merge.set_op(Concatenate(struct, merge, struct.input_nodes))

        vnode1 = VariableNode()
        struct.connect(merge, vnode1)

        vnode1.add_op(Dense(1))

        struct.set_ops([0])

        struct.create_model()

    def test_create_multiple_inputs_with_wrong_output_shape(self):
        from deephyper.search.nas.model.space.struct import DirectStructure
        from deephyper.search.nas.model.space.node import VariableNode
        from deephyper.search.nas.model.space.op.op1d import Dense
        struct = DirectStructure([(5, ), (5, )], (1, ))

        struct.set_ops([])

        with pytest.raises(WrongOutputShape):
            struct.create_model()
