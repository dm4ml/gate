import numpy as np
import pandas as pd
from gate import summarize, detect_drift


def get_df_no_drift(num_cols, num_rows_per_partition=10000):
    # create example date range
    date_range = pd.date_range(start="2022-01-01", periods=30, freq="D")

    # combine data into a DataFrame
    df_elems = []
    for date in date_range:
        date_data = {"date": date}
        date_data = pd.DataFrame(
            {
                "date": [date] * num_rows_per_partition,
                **{
                    f"int_col_{i}": np.random.randint(
                        low=0, high=10, size=num_rows_per_partition
                    )
                    for i in range(num_cols)
                },
                **{
                    f"float_col_{i}": np.random.normal(
                        loc=0, scale=1, size=num_rows_per_partition
                    )
                    for i in range(num_cols)
                },
                **{
                    f"string_col_{i}": np.random.choice(
                        ["A", "B", "C"], size=num_rows_per_partition
                    )
                    for i in range(num_cols)
                },
            }
        )
        df_elems.append(date_data)

    df = pd.concat(df_elems).reset_index(drop=True)

    return df


def test_no_drift_scale():
    import time

    start = time.time()
    df = get_df_no_drift(1000)
    df_creation_time = time.time() - start

    columns = df.columns.to_list()
    columns.remove("date")

    start = time.time()
    summaries = summarize(
        df,
        partition_key="date",
        columns=columns,
    )
    summary_time = time.time() - start

    start = time.time()
    drift_results = detect_drift(summaries[-1], summaries[:-1])
    drift_time = time.time() - start

    print(f"df_creation_time: {df_creation_time}")
    print(f"summary_time: {summary_time}")
    print(f"drift_time: {drift_time}")
    assert False

    assert abs(0.5 - drift_results.score_percentile) < 0.5


def test_drift_scale():
    df = get_df_no_drift(100)

    # Add drift
    max_date = df["date"].max()
    for i in range(50):
        df.loc[df["date"] == max_date, f"int_col_{i}"] = 1000

    columns = df.columns.to_list()
    columns.remove("date")

    summaries = summarize(
        df,
        partition_key="date",
        columns=columns,
    )

    drift_results = detect_drift(summaries[-1], summaries[:-1])

    assert drift_results.score_percentile > 0.9
