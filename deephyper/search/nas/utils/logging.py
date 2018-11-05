import json

class JsonMessage(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        # s = Encoder().encode(self.kwargs)
        s = json.dumps(self.kwargs)
        return '>>> %s' % (s)
