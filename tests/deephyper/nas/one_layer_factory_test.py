import pytest


@pytest.mark.nas
def test_search_space():
    from deephyper.nas.spacelib.tabular import OneLayerSpace

    space = OneLayerSpace(input_shape=(10,), output_shape=(1,)).build()
    model = space.sample()
