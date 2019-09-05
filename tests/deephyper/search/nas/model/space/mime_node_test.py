import tensorflow as tf

from deephyper.search.nas.model.space.node import VariableNode, MimeNode
from deephyper.search.nas.model.space.op.op1d import Dense

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
