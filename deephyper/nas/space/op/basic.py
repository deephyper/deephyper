from .operation import Operation


class Tensor(Operation):
    def __init__(self, tensor, *args, **kwargs):
        self.tensor = tensor

    def __str__(self):
        return str(self.tensor)

    def __call__(self, *args, **kwargs):
        return self.tensor


class Zero(Operation):
    def __init__(self):
        self.tensor = []

    def __str__(self):
        return "Zero"

    def __call__(self, *args, **kwargs):
        return self.tensor
