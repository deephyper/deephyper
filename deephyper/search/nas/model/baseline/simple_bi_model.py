import tensorflow as tf

# from ..space import KSearchSpace
# from ..space.node import VariableNode
# from ..space.op.op1d import Dense, Identity

from deephyper.search.nas.model.space import KSearchSpace
from deephyper.search.nas.model.space.node import VariableNode, ConstantNode
from deephyper.search.nas.model.space.op.op1d import Dense, Identity


def create_search_space(
    input_shape=(100,), output_shape=[(1), (100,)], num_layers=5, **kwargs
):
    struct = KSearchSpace(input_shape, output_shape)

    inp = struct.input_nodes[0]

    # auto-encoder
    units = [128, 64, 32, 16, 8, 16, 32, 64, 128]
    # units = [32, 16, 32]
    prev_node = inp
    d = 1
    for i in range(len(units)):
        vnode = VariableNode()
        vnode.add_op(Identity())
        if d == 1 and units[i] < units[i + 1]:
            d = -1
            # print(min(1, units[i]), ' - ', max(1, units[i])+1)
            for u in range(min(2, units[i]), max(2, units[i]) + 1, 2):
                vnode.add_op(Dense(u, tf.nn.relu))
            latente_space = vnode
        else:
            # print(min(units[i], units[i+d]), ' - ', max(units[i], units[i+d])+1)
            for u in range(
                min(units[i], units[i + d]), max(units[i], units[i + d]) + 1, 2
            ):
                vnode.add_op(Dense(u, tf.nn.relu))
        struct.connect(prev_node, vnode)
        prev_node = vnode

    out2 = ConstantNode(op=Dense(100, name="output_1"))
    struct.connect(prev_node, out2)

    # regressor
    prev_node = latente_space
    # prev_node = inp
    for _ in range(num_layers):
        vnode = VariableNode()
        for i in range(16, 129, 16):
            vnode.add_op(Dense(i, tf.nn.relu))

        struct.connect(prev_node, vnode)
        prev_node = vnode

    out1 = ConstantNode(op=Dense(1, name="output_0"))
    struct.connect(prev_node, out1)

    return struct


def test_create_search_space():
    from random import random, seed
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf

    # seed(10)
    space = create_search_space(num_layers=10)

    # ops = [random() for _ in range(space.num_nodes)]
    ops = [1.0 for _ in range(space.num_nodes)]

    print("num ops: ", len(ops))
    print("ops: ", ops)
    print("size: ", space.size)
    space.set_ops(ops)
    space.draw_graphviz("architecture_baseline.dot")

    model = space.create_model()
    print("depth: ", space.depth)
    plot_model(model, to_file="model_baseline.png", show_shapes=True)
    print("n_parameters: ", model.count_params())

    # model.save('model.h5')


if __name__ == "__main__":
    test_create_search_space()
