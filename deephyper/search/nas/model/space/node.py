from tensorflow.keras.layers import Layer
from deephyper.search.nas.model.space.op.basic import Operation

class Node:
    """This class represents a node of a graph

    Args:
        name (str): node name.
    """

    # Number of 'Node' instances created
    num = 0

    def __init__(self, name='', *args, **kwargs):
        Node.num += 1
        self._num = Node.num
        self._tensor = None
        self.name = name

    def __str__(self):
        return f'{self.name}[id={self._num}]'

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
        elif isinstance(op, Layer):
            return Operation(op)
        else:
            raise RuntimeError(f'Can\'t add this operation \'{op.__name__}\'. An operation should be either of type Operation or Layer when is of type: {type(op)}')


class OperationNode(Node):
    def __init__(self, name='', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def create_tensor(self, inputs=None, train=True, *args, **kwargs):
        if self._tensor is None:
            if inputs == None:
                self._tensor = self.op(train=train)
            else:
                self._tensor = self.op(inputs, train=train)
        return self._tensor


class VariableNode(OperationNode):
    """This class represents a node of a graph where you have a set of possible operations. It means the agent will have to act to choose one of these operations.

    >>> import tensorflow as tf
    >>> from deephyper.search.nas.model.space.node import VariableNode
    >>> vnode = VariableNode("VNode1")
    >>> from deephyper.search.nas.model.space.op.op1d import Dense
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

    def __init__(self, name=''):
        super().__init__(name=name)
        self._ops = list()
        self._index = None

    def __str__(self):
        if self._index != None:
            return f'{super().__str__()}(Variable[{str(self.op)}])'
        else:
            return f'{super().__str__()}(Variable[?])'

    def add_op(self, op):
        self._ops.append(self.verify_operation(op))

    @property
    def num_ops(self):
        return len(self._ops)

    def set_op(self, index):
        self.get_op(index).init()

    def get_op(self, index):
        assert 'float' in str(type(index)) or type(
            index) is int, f'found type is : {type(index)}'
        if 'float' in str(type(index)):
            self._index = self.denormalize(index)
        else:
            assert 0 <= index and index < len(
                self._ops), f'len self._ops: {len(self._ops)}, index: {index}'
            self._index = index
        return self.op

    def denormalize(self, index):
        """Denormalize a normalized index to get an absolute indexes. Useful when you want to compare the number of different architectures.

        Args:
            indexes (float): a normalized index.

        Returns:
            int: An absolute indexes corresponding to the operation choosen with the relative index of `index`.
        """
        assert 0. <= index and index <= 1.
        return int(index * len(self._ops))

    @property
    def op(self):
        if len(self._ops) == 0:
            raise RuntimeError(
                'This VariableNode doesn\'t have any operation yet.')
        elif self._index is None:
            raise RuntimeError(
                'This VariableNode doesn\'t have any set operation, please use "set_op(index)" if you want to set one')
        else:
            return self._ops[self._index]

    @property
    def ops(self):
        return self._ops


class ConstantNode(OperationNode):
    """A ConstantNode represents a node with a fixed operation. It means the agent will not make any new decision for this node. The common use case for this node is to add a tensor in the graph.

    >>> import tensorflow as tf
    >>> from deephyper.search.nas.model.space.node import ConstantNode
    >>> from deephyper.search.nas.model.space.op.op1d import Dense
    >>> cnode = ConstantNode(op=Dense(units=100, activation=tf.nn.relu), name='CNode1')

    Args:
        op (Operation, optional): [description]. Defaults to None.
        name (str, optional): [description]. Defaults to ''.
    """

    def __init__(self, op=None, name='', *args, **kwargs):
        super().__init__(name=name)
        if not op is None:
            op = self.verify_operation(op)
            op.init()  # set operation
        self._op = op

    def set_op(self, op):
        op = self.verify_operation(op)
        op.init()
        self._op = op

    def __str__(self):
        return f'{super().__str__()}(Constant[{str(self.op)}])'

    @property
    def op(self):
        return self._op


class MirrorNode(OperationNode):
    """A MirrorNode is a node which reuse an other, it enable the reuse of keras layers. This node will not add operations to choose.

    Arguments:
        node {Node} -- [description]
    """

    def __init__(self, node):
        super().__init__(name=f"Mirror[{str(node)}]")
        self._node = node

    @property
    def op(self):
        return self._node.op