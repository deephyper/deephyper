import tensorflow as tf

from deephyper.nas.node import VariableNode, MirrorNode, MimeNode
from deephyper.nas.operation import operation

Dense = operation(tf.keras.layers.Dense)


def test_mirror_node():

    vnode = VariableNode()
    vop = Dense(10)
    vnode.add_op(vop)
    vnode.add_op(Dense(20))

    mnode = MirrorNode(vnode)

    vnode.set_op(0)

    assert vnode.op == vop
    assert mnode.op == vop


def test_mime_node():
    vnode = VariableNode()
    vop = Dense(10)
    vnode.add_op(vop)
    vnode.add_op(Dense(20))

    mnode = MimeNode(vnode)
    mop = Dense(30)
    mnode.add_op(mop)
    mnode.add_op(Dense(40))

    vnode.set_op(0)

    assert vnode.op == vop
    assert mnode.op == mop
