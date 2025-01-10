def test_rank():
    from deephyper.analysis import rank

    # Test simple ranking already sorted
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    ranks = [1, 2, 3, 4, 5]
    ranks_ = rank(scores)
    assert all(r_ == r for r_, r in zip(ranks_, ranks)), f"Expected {ranks} but got {ranks_}"

    # Test ranking with ties already sorted
    scores = [0.1, 0.1001, 0.2, 0.3, 0.4, 0.5]
    ranks = [1, 1, 3, 4, 5, 6]
    ranks_ = rank(scores)
    assert all(r_ == r for r_, r in zip(ranks_, ranks)), f"Expected {ranks} but got {ranks_}"

    # Test ranking with ties not sorted
    scores = [0.1, 0.2, 0.1001, 0.3, 0.4, 0.5]
    ranks = [1, 3, 1, 4, 5, 6]
    ranks_ = rank(scores)
    assert all(r_ == r for r_, r in zip(ranks_, ranks)), f"Expected {ranks} but got {ranks_}"
