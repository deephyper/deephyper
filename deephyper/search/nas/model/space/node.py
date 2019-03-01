
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
        return f'{self.name}({self._num})'

    @property
    def id(self):
        return self._num

    @property
    def op(self):
        raise NotImplementedError

    def create_tensor(self, *args, **kwargs):
        raise NotImplementedError

class OperationNode(Node):
    def __init__(self, name='', *args, **kwargs):
        return super().__init__(name=name, *args, **kwargs)

    def create_tensor(self, inputs=None, train=True, *args, **kwargs):
        if self._tensor is None:
            if inputs == None:
                self._tensor = self.op(train=train)
            else:
                self._tensor = self.op(inputs, train=train)
        return self._tensor

class VariableNode(OperationNode):
    """This class represents a node of a graph where you have multiple possible operations.

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
            return f'{super().__str__()}(Variable)'

    def add_op(self, op):
        self._ops.append(op)

    @property
    def num_ops(self):
        return len(self._ops)

    def set_op(self, index):
        self.get_op(index).is_set()

    def get_op(self, index):
        assert 'float' in str(type(index)) or type(index) is int, f'found type is : {type(index)}'
        if 'float' in str(type(index)):
            assert 0. <= index and index <= 1.
            self._index = int(int((index * (len(self._ops) - 1) + 0.5) * 10) / 10)
        else:
            assert 0 <= index and index < len(self._ops), f'len self._ops: {len(self._ops)}, index: {index}'
            self._index = index
        return self.op

    @property
    def op(self):
        if len(self._ops) == 0:
            raise RuntimeError('This VariableNode doesn\'t have any operation yet.')
        elif self._index is None:
            raise RuntimeError('This VariableNode doesn\'t have any set operation, please use "set_op(index)" if you want to set one')
        else:
            return self._ops[self._index]

    @property
    def ops(self):
        return self._ops

class ConstantNode(OperationNode):
    """A ConstantNode is a node which has a fixed operation.

    Arguments:
        op (Operation): operation of the ConstantNode.
    """
    def __init__(self, op=None, name='', *args, **kwargs):
        super().__init__(name=name)
        if not op is None:
            op.is_set() # set operation
        self._op = op

    def set_op(self, op):
        op.is_set()
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