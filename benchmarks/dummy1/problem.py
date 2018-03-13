from collections import OrderedDict
class Problem:
    def __init__(self):
        space = OrderedDict()

        space['x'] = (-5.0, 5.0)
        space['y'] = (-5.0, 5.0)
        space['penalty'] = ['no', 'yes']

        self.space = space
        self.params = self.space.keys()
        self.starting_point = [5.0, 5.0, 'yes']
