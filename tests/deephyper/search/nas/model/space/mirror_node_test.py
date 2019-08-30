from deephyper.search.nas.model.space.node import VariableNode, MirrorNode
from deephyper.search.nas.model.space.op.op1d import Dense

def test_mirror_node():
    vnode = VariableNode()
    vop = Dense(10)
    vnode.add_op(vop)
    vnode.add_op(Dense(20))

    mnode = MirrorNode(vnode)

    vnode.set_op(0)

    assert vnode.op == vop
    assert mnode.op == vop