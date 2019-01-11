from collections import OrderedDict
class Problem:
    def __init__(self):
        space = OrderedDict()

        space['x'] = (-0.3, 0.3)
        space['y'] = (-1.0, 1.0)
        space['penalty'] = ['no', 'yes']

        self.space = space
        self.params = self.space.keys()
        self.starting_point = [0.0, 0.0, 'yes']
