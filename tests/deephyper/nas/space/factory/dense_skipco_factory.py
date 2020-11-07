def test_search_space():
    from deepspace.tabular import DenseSkipCoFactory

    DenseSkipCoFactory().test(input_shape=(10,), output_shape=(1,))


if __name__ == "__main__":
    test_search_space()
