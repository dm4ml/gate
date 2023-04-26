import numpy as np
import pandas as pd
from gate import summarize, detect_drift


# @pytest.mark.flaky(reruns=1)
def test_summarize_embedding_tiny():
    embedding_list = [np.random.rand(3) for _ in range(100)]

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "grp": np.random.choice(["A", "B", "C", "D", "E", "F"], size=100),
            "age": np.random.randint(20, 60, 100),
            "embedding": embedding_list,
        }
    )

    summary = summarize(
        df,
        partition_key="grp",
        columns=["age", "embedding"],
    )
    drift_results = detect_drift(summary[-1], summary[:-1])

    assert not drift_results.is_drifted


# @pytest.mark.flaky(reruns=1)
def test_summarize_embedding_medium():
    embedding_list = [np.random.rand(100) for _ in range(100)]

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "grp": np.random.choice(["A", "B", "C", "D", "E", "F"], size=100),
            "age": np.random.randint(20, 60, 100),
            "embedding": embedding_list,
        }
    )

    summary = summarize(
        df,
        partition_key="grp",
        columns=["age", "embedding"],
    )
    drift_results = detect_drift(summary[-1], summary[:-1])

    assert not drift_results.is_drifted


# @pytest.mark.flaky(reruns=1)
def test_summarize_embedding_big_with_drift():
    embedding_list = [np.random.rand(2048) for _ in range(1000)]
    date_range = pd.date_range(start="2022-01-01", periods=10, freq="D")

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "date": np.random.choice(date_range[:-1], size=1000),
            "embedding": embedding_list,
        }
    )

    prev_summaries = summarize(
        df,
        partition_key="date",
        columns=["embedding"],
    )

    drifted_df = pd.DataFrame(
        {
            "date": [date_range[-1] for _ in range(1000)],
            "embedding": [np.random.rand(2048) * 10 for _ in range(1000)],
        }
    )

    summary = summarize(drifted_df, previous_summaries=prev_summaries)

    drift_results = detect_drift(summary[0], prev_summaries)

    assert drift_results.is_drifted


# def test_clustering_with_drift():
#     num_rows = 1000
#     num_cols = 50
#     data = np.random.rand(num_rows, num_cols)
#     cols = [f"col_{i+1}" for i in range(num_cols)]
#     date_range = pd.date_range(start="2022-01-01", periods=30, freq="D")

#     df = pd.DataFrame(data, columns=cols)
#     df["embedding"] = [np.random.rand(1024) for _ in range(num_rows)]
#     df["embedding2"] = [np.random.rand(1024) + 1 for _ in range(num_rows)]
#     df["date"] = np.random.choice(date_range[:-1], size=num_rows)

#     prev_summaries = summarize(
#         df,
#         partition_key="date",
#         columns=["embedding", "embedding2"] + cols,
#     )

#     drifted_df = pd.DataFrame(data[:1000, :], columns=cols)
#     drifted_df["date"] = [date_range[-1] for _ in range(1000)]
#     drifted_df["embedding"] = [np.random.rand(1024) * 10 for _ in range(1000)]
#     drifted_df["embedding2"] = [
#         np.random.rand(1024) * 10 + 1 for _ in range(1000)
#     ]

#     summary = summarize(drifted_df, previous_summaries=prev_summaries)

#     drift_results = detect_drift(summary[0], prev_summaries)

#     assert drift_results.is_drifted
#     assert (
#         "embedding2" in drift_results.drifted_columns().head(2).index.tolist()
#     )
#     assert (
#         "embedding" in drift_results.drifted_columns().head(2).index.tolist()
#     )
