def test_search_space():
    from deepspace.tabular import DenseSkipCoSpace

    space = DenseSkipCoSpace(input_shape=(10,), output_shape=(1,)).build()
    model = space.sample()



if __name__ == "__main__":
    test_search_space()
