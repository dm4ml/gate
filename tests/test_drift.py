import numpy as np
import pandas as pd
import pytest
from gate import summarize, detect_drift


def test_no_drift(medium_df):
    summary = summarize(
        medium_df,
        partition_column="date",
        columns=["string_col", "int_col", "float_col"],
    )
    assert len(summary) == 30

    drift_results = detect_drift(summary[-1], summary[:-1])

    assert drift_results.score == 0.0


def test_attributes(tiny_df):
    summary = summarize(
        tiny_df,
        partition_column="grp",
        columns=["string_col", "int_col", "float_col"],
    )
    assert len(summary) == 1

    drift_results = detect_drift(summary[-1], summary[:-1])

    assert drift_results.score < 0.01
    assert len(drift_results.all_scores) > 0
    assert drift_results.score_percentile
    assert drift_results.drill_down().sum().values[0] < 0.01

    expected_result = pd.DataFrame(
        [
            {"statistic": "coverage", "z-score": 0.0},
            {"statistic": "coverage", "z-score": 0.0},
            {"statistic": "occurrence_ratio", "z-score": 0.0},
        ]
    )

    assert (
        drift_results.drifted_columns()
        .reset_index(drop=True)
        .equals(expected_result)
    )


def test_drift(df_with_drift):
    summary = summarize(
        df_with_drift,
        partition_column="date",
        columns=["string_col", "int_col", "float_col"],
    )
    assert len(summary) == 10

    drift_results = detect_drift(summary[-1], summary[:-1])

    assert drift_results.score_percentile > 0.9

    assert drift_results.drifted_columns().index.values[0] == "int_col"
    assert drift_results.drifted_columns()["z-score"].abs().values[0] > 2.0


def test_hello(df_with_drift):
    columns = df_with_drift.columns.tolist()
    columns.remove("date")
    summary = summarize(
        df_with_drift,
        partition_column="date",
        columns=columns,
    )

    drift_results = detect_drift(summary[-1], summary[:-1], cluster=True)

    assert False
