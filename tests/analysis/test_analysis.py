from pathlib import Path

import numpy as np


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


# parameters_at_topk,
# plot_search_trajectory_single_objective_hpo,
# plot_worker_utilization,


def test_analysis_hpo_read_results(tmp_path):
    from deephyper.analysis.hpo import (
        filter_failed_objectives,
        get_mask_of_rows_without_failures,
        read_results_from_csv,
        parameters_from_row,
        parameters_at_max,
        parameters_at_topk,
    )

    csv_path = Path(tmp_path) / "results.csv"
    with open(csv_path, "w") as f:
        f.write(
            "p:x,objective,job_id,job_status,m:timestamp_submit,m:timestamp_gather\n"
            "0.5,0.5,0,DONE,0.1,0.2\n"
            "0.2,F,1,DONE,0.1,0.2\n"
            "0.8,0.9,2,DONE,0.1,0.2\n"
            "0.9,0.9,3,DONE,0.1,0.2\n"
        )

    df = read_results_from_csv(csv_path)
    assert len(df) == 4

    df_success, df_failed = filter_failed_objectives(df)
    assert len(df_success) == 3
    assert len(df_failed) == 1

    has_any_failure, mask_no_failures = get_mask_of_rows_without_failures(df, column="objective")
    assert has_any_failure
    assert np.all(mask_no_failures == np.array([1, 0, 1, 1], dtype=bool))

    has_any_failure, mask_no_failures = get_mask_of_rows_without_failures(
        df_success, column="objective"
    )
    assert not has_any_failure
    assert np.all(mask_no_failures == np.array([1, 1, 1], dtype=bool))

    params = parameters_from_row(df.iloc[0])
    assert len(params) == 1
    assert params["x"] == 0.5

    params, objective = parameters_at_max(df)
    assert len(params) == 1
    assert params["x"] == 0.9
    assert objective == 0.9

    topk_results = parameters_at_topk(df, k=3)
    assert len(topk_results) == 3
    assert len(topk_results[0][0]) == 1
    assert topk_results[0][0]["x"] == 0.9
    assert topk_results[0][1] == 0.9

    assert len(topk_results[1][0]) == 1
    assert topk_results[1][0]["x"] == 0.8
    assert topk_results[1][1] == 0.9

    assert len(topk_results[2][0]) == 1
    assert topk_results[2][0]["x"] == 0.5
    assert topk_results[2][1] == 0.5
