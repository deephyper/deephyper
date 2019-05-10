import numpy as np
import json


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif type(obj) in [np.int32, np.int64]:
            return int(obj)
        elif type(obj) in [np.float32, np.float64]:
            return float(obj)
        else:
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)


class JsonMessage(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        s = json.dumps(self.kwargs, cls=Encoder)
        return '>>> %s' % (s)


def test_ndarray():
    t = np.array([1, 2])
    jm = JsonMessage(t=t)
    print(jm)


if __name__ == '__main__':
    test_ndarray()
