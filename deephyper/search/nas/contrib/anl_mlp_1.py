from deephyper.search.nas.cell.mlp import create_dense_cell_type1
from deephyper.search.nas.cell.structure import create_sequential_structure


def create_structure(input_tensor, num_cells):
    return create_sequential_structure(input_tensor, create_dense_cell_type1, num_cells)

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    from random import random

    inpt = tf.constant(np.zeros((2, 28)), dtype=tf.float32)
    num_cells = 5
    net_struct = create_structure(inpt, num_cells)

    cell_ops = [random() for _ in range(net_struct.num_nodes_cell(i=0)[0])]
    ops = cell_ops * num_cells
    print(f'ops: {ops}')
    net_struct.set_ops(ops)
    net_struct.draw_graphviz('anl_mlp_1.dot')

    out = net_struct.create_tensor()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        res = sess.run(out)

        print(f'res: {res}')
        print(f'input shape: {inpt.get_shape()}')
        print(f'outpt shape: {np.shape(res)}')
