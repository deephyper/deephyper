import numpy as np
import pytest


def wrap_and_predict(module):
    from deephyper.predictor.torch import TorchPredictor

    predictor = TorchPredictor(module)

    x = np.zeros((16, 1), dtype=np.float32)
    y = predictor.predict(x)
    return y


@pytest.mark.torch
def test_torch_predictor_with_single_output():
    import torch.nn as nn

    class DummyNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )

        def forward(self, x):
            return self.net(x)

    y = wrap_and_predict(DummyNN())
    assert isinstance(y, np.ndarray)
    assert np.shape(y) == (16, 1)


@pytest.mark.torch
def test_torch_predictor_with_list_output():
    import torch.nn as nn

    class DummyNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )

        def forward(self, x):
            out = self.net(x)
            return [out, out]

    y = wrap_and_predict(DummyNN())

    assert type(y) is list
    assert len(y) == 2
    assert all(isinstance(y[i], np.ndarray) for i in range(2))
    assert all(np.shape(y[i]) == (16, 1) for i in range(2))


@pytest.mark.torch
def test_torch_predictor_with_dict_output():
    import torch.nn as nn

    class DummyNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )

        def forward(self, x):
            out = self.net(x)
            return {"output_0": out, "output_1": out}

    y = wrap_and_predict(DummyNN())

    assert type(y) is dict
    assert len(y) == 2
    assert all(isinstance(v, np.ndarray) for v in y.values())
    assert all(np.shape(v) == (16, 1) for v in y.values())


@pytest.mark.torch
def test_torch_predictor_with_proba():
    import torch.nn as nn

    class DummyNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 2),
            )
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            return self.net(x)

        def predict_proba(self, x):
            return self.softmax(self.forward(x))

    y = wrap_and_predict(DummyNN())

    assert isinstance(y, np.ndarray)
    assert np.shape(y) == (16, 2)
    assert np.all((0 <= y) & (y <= 1))


if __name__ == "__main__":
    test_torch_predictor_with_proba()
