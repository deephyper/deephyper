from collections import OrderedDict
import random
NDIM = 30

class Problem:
    def __init__(self):
        space = OrderedDict()

        for i in range(1, 1+NDIM):
            dim = f"x{i}"
            space[dim] = (-3.0, 4.0)

        self.space = space
        self.params = self.space.keys()
        self.starting_point = [random.uniform(-3.0, 4.0) for i in range(NDIM)]

if __name__ == "__main__":
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))
