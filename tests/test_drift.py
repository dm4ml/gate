import pytest
from gate import summarize, detect_drift


def test_no_drift(medium_df):
    summary = summarize(
        medium_df,
        partition_key="date",
        columns=["string_col", "int_col", "float_col"],
    )
    assert len(summary) == 30

    drift_results = detect_drift(summary[-1], summary[:-1])

    assert drift_results.score < 1e-7


def test_attributes(tiny_df):
    summary = summarize(
        tiny_df,
        partition_key="grp",
        columns=["string_col", "int_col", "float_col"],
    )
    assert len(summary) == 1

    with pytest.raises(ValueError):
        detect_drift(summary[-1], summary[:-1])


def test_drift(df_with_drift):
    summary = summarize(
        df_with_drift,
        partition_key="date",
        columns=["string_col", "int_col", "float_col"],
    )
    assert len(summary) == 10

    drift_results = detect_drift(summary[-1], summary[:-1])

    assert drift_results.score_percentile > 0.85

    assert drift_results.drifted_columns().index.values[0] in [
        "int_col",
        "float_col",
    ]
    assert drift_results.drifted_columns()["z-score"].abs().values[0] > 2.0


def test_drift_small_clustering(df_with_drift):
    columns = df_with_drift.columns.tolist()
    columns.remove("date")
    summary = summarize(
        df_with_drift,
        partition_key="date",
        columns=columns,
    )

    drift_results = detect_drift(summary[-1], summary[:-1], cluster=True)

    assert len(drift_results.clustering) > 0
    assert drift_results.score_percentile > 0.85
    assert drift_results.drifted_columns().index.values[0] in [
        "int_col",
        "float_col",
    ]

    assert len(drift_results.drifted_columns()) > 3
