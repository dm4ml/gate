import pytest
import pandas as pd
import random
import numpy as np


@pytest.fixture
def tiny_df():
    df = pd.DataFrame(
        {
            "grp": ["A"] * 3,
            "string_col": ["cat", "dog", "dog"],
            "int_col": [0, 1, None],
            "float_col": [0.0, 0.1, 0.2],
        }
    )
    return df


@pytest.fixture
def small_df():
    # create example data
    groups = ["A", "B", "C", "D", "E"] * 2
    string_values = ["foo", "bar", "baz"] * 3
    string_values.append(None)
    int_values = [random.randint(0, 100) for _ in range(10)]
    float_values = [random.uniform(0, 1) for _ in range(10)]

    # create DataFrame
    df = pd.DataFrame(
        {
            "grp": groups,
            "string_col": string_values,
            "int_col": int_values,
            "float_col": float_values,
        }
    )
    return df


@pytest.fixture
def medium_df():
    # create example date range
    date_range = pd.date_range(start="2022-01-01", periods=30, freq="D")

    # create example data for each column
    int_col = np.random.randint(low=0, high=10, size=10000)
    float_col = np.random.normal(loc=0, scale=1, size=10000)
    string_col = np.random.choice(["A", "B", "C"], size=10000)

    # combine data into a DataFrame
    df_elems = []
    for date in date_range:
        date_data = {"date": date}
        date_data = pd.DataFrame(
            {
                "date": [date] * len(int_col),
                "int_col": int_col,
                "float_col": float_col,
                "string_col": string_col,
            }
        )
        df_elems.append(date_data)

    df = pd.concat(df_elems).reset_index(drop=True)
    return df


@pytest.fixture
def df_with_drift():
    # create example date range
    date_range = pd.date_range(start="2022-01-01", periods=10, freq="D")

    # combine data into a DataFrame
    df_elems = []
    for date in date_range:
        if date != date_range[-1]:
            int_col = np.random.randint(low=0, high=10, size=10000)
            float_col = np.random.normal(loc=0, scale=1, size=10000)
            string_col = np.random.choice(["A", "B", "C"], size=10000)
        else:
            int_col = np.random.randint(low=10, high=20, size=10000)
            float_col = np.random.normal(loc=1, scale=2, size=10000)
            string_col = np.random.choice(["D", "B", "C"], size=10000)

        date_data = {"date": date}
        date_data = pd.DataFrame(
            {
                "date": [date] * len(int_col),
                "int_col": int_col,
                "float_col": float_col,
                "string_col": string_col,
            }
        )
        df_elems.append(date_data)

    df = pd.concat(df_elems).reset_index(drop=True)
    return df
