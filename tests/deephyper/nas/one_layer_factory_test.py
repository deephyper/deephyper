def test_search_space():
    from deepspace.tabular import OneLayerSpace

    space = OneLayerSpace(input_shape=(10,), output_shape=(1,)).build()
    model = space.sample()

