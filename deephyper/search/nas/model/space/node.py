import tensorflow as tf

from deephyper.search.nas.model.space.op.basic import Connect


class Node:
    """This class represents a node of a graph.

    Args:
        name (str): node name.
        ops (list): possible operations of node.
        index (int): index corresponding to the operation choosen for this node among the possible operations
    """

    num = 0

    def __init__(self, name='', ops=[], index=None):
        Node.num += 1
        self._num = Node.num
        self.name = name
        self._ops = ops[:]
        assert index is None or (0 <= index and index < len(self._ops))
        self._index = index
        self._tensor = None

    @property
    def id(self):
        return self._num

    def __str__(self):
        if self._index != None:
            return f'{self.name}({self._num})[{str(self._ops[self._index])}]'
        else:
            return f'{self.name}({self._num})'

    def add_op(self, op):
        self._ops.append(op)

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
        return self._ops[self._index]

    def create_tensor(self, inputs=None, train=True):
        assert self._index != None

        if self._tensor is None:
            if inputs == None:
                self._tensor = self._ops[self._index](train=train)
            else:
                self._tensor = self._ops[self._index](inputs, train=train)
        return self._tensor

    def get_ops(self):
        return self._ops

if __name__ == '__main__':
    n = Node()
