class Node:
    num = 0

    def __init__(self, name="", *args, **kwargs):
        Node.num += 1
        self._num = Node.num
        self.name = name

    def __str__(self):
        return f"{self.name}"

    @property
    def id(self):
        return self._num

    @property
    def op(self):
        raise NotImplementedError

class InputNode(Node):
    def __init__(self):
        super().__init__("input")

class OutputNode(Node):
    def __init__(self):
        super().__init__("output")

class OperationNode(Node):
    def __init__(self, name="", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

class VariableNode(OperationNode):
    def __init__(self, name=""):
        super().__init__(name=name)
        self._ops = list()
        self._index = None

    def __str__(self):
        if self._index != None:
            return f"{str(self.op)}"
        else:
            return f"?"

    def add_op(self, op):
        self._ops.append(op)

    @property
    def num_ops(self):
        return len(self._ops)

    def set_op(self, index):
        self.get_op(index).init(self)

    def get_op(self, index):
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
    def __init__(self, op=None):
        super().__init__(name=str(op))
        if not op is None:
            op.init(self)  # set operation
        self._op = op

    def set_op(self, op):
        self.name = str(op)
        op.init(self)
        self._op = op

    @property
    def op(self):
        return self._op