import pytest
from deephyper.core.exceptions.nas.space import WrongOutputShape


@pytest.mark.incremental
class TestKSearchSpace:
    def test_import(self):
        from deephyper.nas import KSearchSpace

    def test_create(self):
        from deephyper.nas import KSearchSpace

        KSearchSpace((5,), (1,))

    def test_create_one_vnode(self):
        from deephyper.nas import KSearchSpace

        struct = KSearchSpace((5,), (1,))

        from deephyper.nas.node import VariableNode

        vnode = VariableNode()

        struct.connect(struct.input_nodes[0], vnode)

        import tensorflow as tf
        from deephyper.nas.operation import operation

        Dense = operation(tf.keras.layers.Dense)

        vnode.add_op(Dense(1))

        struct.set_ops([0])

        falias = "test_keras_search_spaceure"
        struct.draw_graphviz(f"{falias}.dot")

        model = struct.create_model()
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=f"{falias}.png", show_shapes=True)

    def test_create_more_nodes(self):
        from deephyper.nas import KSearchSpace
        from deephyper.nas.node import VariableNode

        import tensorflow as tf
        from deephyper.nas.operation import operation

        Dense = operation(tf.keras.layers.Dense)

        struct = KSearchSpace((5,), (1,))

        vnode1 = VariableNode()
        struct.connect(struct.input_nodes[0], vnode1)

        vnode1.add_op(Dense(10))

        vnode2 = VariableNode()
        vnode2.add_op(Dense(1))

        struct.connect(vnode1, vnode2)

        struct.set_ops([0, 0])

        falias = "test_keras_search_spaceure"
        struct.draw_graphviz(f"{falias}.dot")

        model = struct.create_model()
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=f"{falias}.png", show_shapes=True)

    def test_create_multiple_inputs_with_one_vnode(self):
        import tensorflow as tf
        from deephyper.nas import KSearchSpace
        from deephyper.nas.node import VariableNode, ConstantNode
        from deephyper.nas.operation import operation, Concatenate

        Dense = operation(tf.keras.layers.Dense)

        struct = KSearchSpace([(5,), (5,)], (1,))

        merge = ConstantNode()
        merge.set_op(Concatenate(struct, struct.input_nodes))

        vnode1 = VariableNode()
        struct.connect(merge, vnode1)

        vnode1.add_op(Dense(1))

        struct.set_ops([0])

        struct.create_model()