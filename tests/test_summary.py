import numpy as np
import pandas as pd
import pytest
from gate import summarize


def test_summarize(tiny_df):
    summary = summarize(
        tiny_df,
        partition_key="grp",
        columns=["string_col", "int_col", "float_col"],
    )
    assert len(summary) == 1
    summary = summary[0].value

    assert len(summary) == 3

    # Check all the statistics

    expected_result = pd.DataFrame(
        [
            {
                "grp": "A",
                "column": "float_col",
                "coverage": 1.0,
                "mean": 0.10000000000000002,
                "num_unique_values": np.nan,
                "occurrence_ratio": np.nan,
                "p50": 0.1,
                "p95": 0.2,
            },
            {
                "grp": "A",
                "column": "int_col",
                "coverage": 0.6666666666666666,
                "mean": 0.5,
                "num_unique_values": np.nan,
                "occurrence_ratio": np.nan,
                "p50": 1.0,
                "p95": 1.0,
            },
            {
                "grp": "A",
                "column": "string_col",
                "coverage": 1.0,
                "mean": np.nan,
                "num_unique_values": 2.0,
                "occurrence_ratio": 0.6666666666666666,
                "p50": np.nan,
                "p95": np.nan,
            },
        ]
    )

    assert expected_result.equals(summary)


def test_bad_df(tiny_df, tiny_df_2):
    summary = summarize(
        tiny_df,
        partition_key="grp",
        columns=["string_col", "int_col", "float_col"],
    )

    with pytest.raises(ValueError):
        summarize(tiny_df_2, previous_summaries=summary)
