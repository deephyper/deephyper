class Operation:
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        try:
            return str(self)
        except:
            return type(self).__name__

    def init(self, current_node):
        pass

class Conv3X3(Operation):
    def __str__(self):
        return 'conv3x3-bn-relu'

class Conv1X1(Operation):
    def __str__(self):
        return 'conv1x1-bn-relu'

class MaxPool3X3(Operation):
    def __str__(self):
        return 'maxpool3x3'


class Identity(Operation):
    def __str__(self):
        return 'id'

class Connect(Operation):
    def __init__(self, search_space, source_node):
        self.search_space = search_space
        self.source_node = source_node
        self.destin_node = None

    def __str__(self):
        return "connect"

    def init(self, current_node):
        """Set the connection in the search_space graph from [n1] -> n2.
        """
        self.destin_node = current_node
        self.search_space.connect(self.source_node, self.destin_node)
