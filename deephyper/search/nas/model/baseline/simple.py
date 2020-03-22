import tensorflow as tf

from ..space import AutoKSearchSpace
from ..space.node import VariableNode, ConstantNode
from ..space.op.op1d import Dense
from ..space.op.merge import Concatenate

def gen_vnode():
    vnode1 = VariableNode()
    for i in range(1, 11):
        vnode1.add_op(Dense(i, tf.nn.relu))
    return vnode1

def create_search_space(input_shape=(2,), output_shape=(1,), **kwargs):
    ss = AutoKSearchSpace(input_shape, output_shape, regression=True)

    if type(input_shape) is list:
        vnodes = []
        for i in range(len(input_shape)):
            vn = gen_vnode()
            vnodes.append(vn)
            ss.connect(ss.input_nodes[i], vn)

        cn = ConstantNode()
        cn.set_op(Concatenate(ss, vnodes))

        vn = gen_vnode()
        ss.connect(cn,vn)

    else:
        vnode1 = gen_vnode()
        ss.connect(ss.input_nodes[0], vnode1)

    return ss