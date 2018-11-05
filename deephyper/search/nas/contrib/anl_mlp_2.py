from deephyper.search.nas.cell.mlp import create_dense_cell_type2
from deephyper.search.nas.cell.structure import create_seq_struct_full_skipco


def create_structure(input_tensor, num_cells):
    return create_seq_struct_full_skipco(input_tensor, create_dense_cell_type2, num_cells)

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    from random import random

    inpt = tf.constant(np.zeros((2, 28)), dtype=tf.float32)
    num_cells = 5
    net_struct = create_structure(inpt, num_cells)

    ops = [random() for _ in range(net_struct.num_nodes)]
    print(f'ops: {ops}')
    print(f'num ops: {len(ops)}')

    net_struct.set_ops(ops)
    net_struct.draw_graphviz('anl_mlp_2.dot')

    out = net_struct.create_tensor()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        res = sess.run(out)

        print(f'res: {res}')
        print(f'input shape: {inpt.get_shape()}')
        print(f'outpt shape: {np.shape(res)}')
