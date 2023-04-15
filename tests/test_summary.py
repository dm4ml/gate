import numpy as np
import pandas as pd
import pytest
from gate import summarize


def test_summarize(tiny_df):
    summary = summarize(
        tiny_df,
        partition_column="grp",
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
                "num_frequent_values": 1.0,
                "num_unique_values": np.nan,
                "occurrence_ratio": 0.3333333333333333,
                "stdev": 0.08164965809277261,
            },
            {
                "grp": "A",
                "column": "int_col",
                "coverage": 0.6666666865348816,
                "mean": 0.5,
                "num_frequent_values": 1.0,
                "num_unique_values": np.nan,
                "occurrence_ratio": 0.5,
                "stdev": 0.5,
            },
            {
                "grp": "A",
                "column": "string_col",
                "coverage": 1.0,
                "mean": np.nan,
                "num_frequent_values": np.nan,
                "num_unique_values": 2.0,
                "occurrence_ratio": 0.6666666666666666,
                "stdev": np.nan,
            },
        ]
    )

    assert expected_result.equals(summary)
