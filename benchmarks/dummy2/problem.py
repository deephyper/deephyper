from collections import OrderedDict

class Problem():
    def __init__(self):
        space = OrderedDict()
        space['epochs'] = (2, 500)
        space['learning_rate'] = (1e-04, 0.05)
        space['penalty'] = (0.0, 3.0)

        self.space = space
        self.params = self.space.keys()
        self.starting_point = [10, 0.01, 0.5]
