import pytest


@pytest.mark.incremental
class TestAutoKSearchSpace:
    def test_import(self):
        from deephyper.nas import AutoKSearchSpace

    def test_create(self):
        from deephyper.nas import AutoKSearchSpace

        AutoKSearchSpace((5,), (1,), regression=True)

    def test_create_one_vnode(self):
        import tensorflow as tf
        from deephyper.nas import AutoKSearchSpace
        from deephyper.nas.operation import operation
        from deephyper.nas.node import VariableNode

        Dense = operation(tf.keras.layers.Dense)

        struct = AutoKSearchSpace((5,), (1,), regression=True)

        vnode = VariableNode()

        struct.connect(struct.input_nodes[0], vnode)

        vnode.add_op(Dense(10))

        struct.set_ops([0])

        falias = "test_auto_keras_search_spaceure"
        struct.draw_graphviz(f"{falias}.dot")

        model = struct.create_model()
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=f"{falias}.png", show_shapes=True)

    def test_create_more_nodes(self):
        import tensorflow as tf
        from deephyper.nas import AutoKSearchSpace
        from deephyper.nas.operation import operation
        from deephyper.nas.node import VariableNode

        Dense = operation(tf.keras.layers.Dense)

        struct = AutoKSearchSpace((5,), (1,), regression=True)

        vnode1 = VariableNode()
        struct.connect(struct.input_nodes[0], vnode1)

        vnode1.add_op(Dense(10))

        vnode2 = VariableNode()
        vnode2.add_op(Dense(10))

        struct.connect(vnode1, vnode2)

        struct.set_ops([0, 0])

        falias = "test_auto_keras_search_spaceure"
        struct.draw_graphviz(f"{falias}.dot")

        model = struct.create_model()
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=f"{falias}.png", show_shapes=True)

    def test_create_multiple_inputs(self):
        import tensorflow as tf
        from deephyper.nas import AutoKSearchSpace
        from deephyper.nas.operation import operation
        from deephyper.nas.node import VariableNode

        Dense = operation(tf.keras.layers.Dense)

        struct = AutoKSearchSpace([(5,), (5,)], (1,), regression=True)

        struct.set_ops([])

        falias = "test_auto_keras_search_spaceure"
        struct.draw_graphviz(f"{falias}.dot")

        model = struct.create_model()
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=f"{falias}.png", show_shapes=True)
