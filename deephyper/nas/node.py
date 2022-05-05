"""This module provides the available node types to build a ``KSearchSpace``. 
"""
import tensorflow as tf

import deephyper.core.exceptions
from deephyper.nas.operation import Operation


class Node:
    """Represents a node of a ``KSearchSpace``.

    Args:
        name (str): node name.
    """

    # Number of 'Node' instances created
    num = 0

    def __init__(self, name="", *args, **kwargs):
        Node.num += 1
        self._num = Node.num
        self._tensor = None
        self.name = name

    def __str__(self):
        return f"{self.name}[id={self._num}]"

    @property
    def id(self):
        return self._num

    @property
    def op(self):
        raise NotImplementedError

    def create_tensor(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def verify_operation(op):
        if isinstance(op, Operation):
            return op
        elif isinstance(op, tf.keras.layers.Layer):
            return Operation(op)
        else:
            raise RuntimeError(
                f"Can't add this operation '{op.__name__}'. An operation should be either of type Operation or tf.keras.layers.Layer when is of type: {type(op)}"
            )


class OperationNode(Node):
    def __init__(self, name="", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def create_tensor(self, inputs=None, train=True, seed=None, **kwargs):
        if self._tensor is None:
            if inputs is None:
                try:
                    self._tensor = self.op(train=train, seed=None)
                except TypeError:
                    raise RuntimeError(
                        f'Verify if node: "{self}" has incoming connexions!'
                    )
            else:
                self._tensor = self.op(inputs, train=train)
        return self._tensor


class VariableNode(OperationNode):
    """This class represents a node of a graph where you have a set of possible operations. It means the agent will have to act to choose one of these operations.

    >>> import tensorflow as tf
    >>> from deephyper.nas.space.node import VariableNode
    >>> vnode = VariableNode("VNode1")
    >>> from deephyper.nas.space.op.op1d import Dense
    >>> vnode.add_op(Dense(
    ... units=10,
    ... activation=tf.nn.relu))
    >>> vnode.num_ops
    1
    >>> vnode.add_op(Dense(
    ... units=1000,
    ... activation=tf.nn.tanh))
    >>> vnode.num_ops
    2
    >>> vnode.set_op(0)
    >>> vnode.op.units
    10

    Args:
        name (str): node name.
    """

    def __init__(self, name=""):
        super().__init__(name=name)
        self._ops = list()
        self._index = None

    def __str__(self):
        if self._index != None:
            return f"{super().__str__()}(Variable[{str(self.op)}])"
        else:
            return f"{super().__str__()}(Variable[?])"

    def add_op(self, op):
        self._ops.append(self.verify_operation(op))

    @property
    def num_ops(self):
        return len(self._ops)

    def set_op(self, index):
        self.get_op(index).init(self)

    def get_op(self, index):
        assert "float" in str(type(index)) or "int" in str(
            type(index)
        ), f"found type is : {type(index)}"
        if "float" in str(type(index)):
            self._index = self.denormalize(index)
        else:
            assert 0 <= index and index < len(
                self._ops
            ), f"Number of possible operations is: {len(self._ops)}, but index given is: {index} (index starts from 0)!"
            self._index = index
        return self.op

    def denormalize(self, index):
        """Denormalize a normalized index to get an absolute indexes. Useful when you want to compare the number of different search_spaces.

        Args:
            indexes (float|int): a normalized index.

        Returns:
            int: An absolute indexes corresponding to the operation choosen with the relative index of `index`.
        """
        if type(index) is int:
            return index
        else:
            assert 0.0 <= index and index <= 1.0
            res = int(index * len(self._ops))
            if index == 1.0:
                res -= 1
            return res

    @property
    def op(self):
        if len(self._ops) == 0:
            raise RuntimeError("This VariableNode doesn't have any operation yet.")
        elif self._index is None:
            raise RuntimeError(
                'This VariableNode doesn\'t have any set operation, please use "set_op(index)" if you want to set one'
            )
        else:
            return self._ops[self._index]

    @property
    def ops(self):
        return self._ops


class ConstantNode(OperationNode):
    """A ConstantNode represents a node with a fixed operation. It means the agent will not make any new decision for this node. The common use case for this node is to add a tensor in the graph.

    >>> import tensorflow as tf
    >>> from deephyper.nas.space.node import ConstantNode
    >>> from deephyper.nas.space.op.op1d import Dense
    >>> cnode = ConstantNode(op=Dense(units=100, activation=tf.nn.relu), name='CNode1')
    >>> cnode.op
    Dense_100_relu

    Args:
        op (Operation, optional): operation to fix for this node. Defaults to None.
        name (str, optional): node name. Defaults to ``''``.
    """

    def __init__(self, op=None, name="", *args, **kwargs):
        super().__init__(name=name)
        if not op is None:
            op = self.verify_operation(op)
            op.init(self)  # set operation
        self._op = op

    def set_op(self, op):
        op = self.verify_operation(op)
        op.init(self)
        self._op = op

    def __str__(self):
        return f"{super().__str__()}(Constant[{str(self.op)}])"

    @property
    def op(self):
        return self._op


class MirrorNode(OperationNode):
    """A MirrorNode is a node which reuse an other, it enable the reuse of tf.keras layers. This node will not add operations to choose.

    Args:
        node (Node): The targeted node to mirror.

    >>> from deephyper.nas.space.node import VariableNode, MirrorNode
    >>> from deephyper.nas.space.op.op1d import Dense
    >>> vnode = VariableNode()
    >>> vnode.add_op(Dense(10))
    >>> vnode.add_op(Dense(20))
    >>> mnode = MirrorNode(vnode)
    >>> vnode.set_op(0)
    >>> vnode.op
    Dense_10
    >>> mnode.op
    Dense_10
    """

    def __init__(self, node):

        super().__init__(name=f"Mirror[{str(node)}]")
        self._node = node

    @property
    def op(self):
        return self._node.op


class MimeNode(OperationNode):
    """A MimeNode is a node which reuse an the choice made for an VariableNode, it enable the definition of a Cell based search_space. This node reuse the operation from the mimed VariableNode but only the choice made.

    Args:
        node (VariableNode): the VariableNode to mime.

    >>> from deephyper.nas.space.node import VariableNode, MimeNode
    >>> from deephyper.nas.space.op.op1d import Dense
    >>> vnode = VariableNode()
    >>> vnode.add_op(Dense(10))
    >>> vnode.add_op(Dense(20))
    >>> mnode = MimeNode(vnode)
    >>> mnode.add_op(Dense(30))
    >>> mnode.add_op(Dense(40))
    >>> vnode.set_op(0)
    >>> vnode.op
    Dense_10
    >>> mnode.op
    Dense_30
    """

    def __init__(self, node, name=""):
        super().__init__(name=f"Mime[{name}][src={str(node)}]")
        self.node = node
        self._ops = list()

    def add_op(self, op):
        self._ops.append(self.verify_operation(op))

    @property
    def num_ops(self):
        return len(self._ops)

    def set_op(self):
        if self.node._index is None:
            raise deephyper.core.exceptions.DeephyperRuntimeError(
                f"{str(self)} cannot be initialized because its source {str(self.node)} is not initialized!"
            )
        self._ops[self.node._index].init(self)

    @property
    def op(self):
        if self.num_ops != self.node.num_ops:
            raise deephyper.core.exceptions.DeephyperRuntimeError(
                f"{str(self)} and {str(self.node)} should have the same number of opertions, when {str(self)} has {self.num_ops} and {str(self.node)} has {self.node.num_ops}!"
            )
        else:
            return self._ops[self.node._index]

    @property
    def ops(self):
        return self._ops
