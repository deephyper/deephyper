def test_search_space():
    from deepspace.tabular import OneLayerFactory

    OneLayerFactory().test(input_shape=(10,), output_shape=(1,))

