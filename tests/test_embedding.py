import numpy as np
import pandas as pd
from gate import summarize, detect_drift, compute_embeddings

import os
import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.flaky(reruns=2)
def test_summarize_embedding_tiny():
    embedding_list = [np.random.rand(3) for _ in range(100)]

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "grp": np.random.choice(["A", "B", "C", "D", "E", "F"], size=100),
            "embedding_key": np.random.randint(20, 60, 100),
            "embedding_value": embedding_list,
        }
    )

    summary = summarize(
        df,
        partition_key="grp",
        embedding_column_map={"embedding_key": "embedding_value"},
    )
    drift_results = detect_drift(summary[-1], summary[:-1])

    assert not drift_results.is_drifted


@pytest.mark.flaky(reruns=2)
def test_summarize_embedding_medium():
    embedding_list = [np.random.rand(100) for _ in range(100)]

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "grp": np.random.choice(["A", "B", "C", "D", "E", "F"], size=100),
            "embedding_key": np.random.randint(20, 60, 100),
            "embedding_value": embedding_list,
        }
    )

    summary = summarize(
        df,
        partition_key="grp",
        embedding_column_map={"embedding_key": "embedding_value"},
    )
    drift_results = detect_drift(summary[-1], summary[:-1])

    assert not drift_results.is_drifted


@pytest.mark.flaky(reruns=2)
def test_summarize_embedding_big_with_drift():
    embedding_list = [np.random.rand(2048) for _ in range(1000)]
    date_range = pd.date_range(start="2022-01-01", periods=10, freq="D")

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "date": np.random.choice(date_range[:-1], size=1000),
            "embedding_key": np.random.choice(
                np.random.randint(20, 60, 100), size=1000
            ),
            "embedding_value": embedding_list,
        }
    )

    prev_summaries = summarize(
        df,
        partition_key="date",
        embedding_column_map={"embedding_key": "embedding_value"},
    )

    drifted_df = pd.DataFrame(
        {
            "date": [date_range[-1] for _ in range(1000)],
            "embedding_key": np.random.choice(
                np.random.randint(20, 60, 100), size=1000
            ),
            "embedding_value": [
                np.random.rand(2048) * 10 for _ in range(1000)
            ],
        }
    )

    summary = summarize(drifted_df, previous_summaries=prev_summaries)

    assert len(summary[0].embedding_examples("embedding_key")) > 0
    assert len(summary[0].embedding_examples("embedding_key").columns) == 4

    with pytest.raises(ValueError):
        summary[0].embedding_examples("nonexsistent_key")

    drift_results = detect_drift(summary[0], prev_summaries)

    assert drift_results.is_drifted

    examples = drift_results.drifted_examples("embedding_key")
    assert "drifted_examples" in examples.keys()
    assert "corresponding_examples" in examples.keys()


def test_compute_embedding():
    # Create dataframe with image urls from the internet
    df = pd.DataFrame(
        {
            "url": [
                "https://picsum.photos/200/300/?random&{}".format(i)
                for i in range(1, 11)
            ],
            "id": [i for i in range(1, 11)],
        }
    )
    embeddings = compute_embeddings(df["url"], column_type="image")
    assert len(embeddings) == 10


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="takes too long")
def test_image_embedding_drift():
    # Create dataframe with image urls from the internet
    date_range = pd.date_range(start="2022-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "url": [
                "https://picsum.photos/200/300/?random&{}".format(i)
                for i in range(1, 51)
            ],
            "date": np.random.choice(date_range[:-1], size=50),
        }
    )
    df["embedding"] = compute_embeddings(df["url"], column_type="image")

    prev_summaries = summarize(
        df,
        partition_key="date",
        embedding_column_map={"url": "embedding"},
    )

    drifted_df = pd.DataFrame(
        {
            "date": [date_range[-1] for _ in range(10)],
            "url": ["fake_url_{}".format(i) for i in range(1, 11)],
            "embedding": [np.random.rand(512) for _ in range(10)],
        }
    )

    summary = summarize(drifted_df, previous_summaries=prev_summaries)[0]

    assert len(summary.embedding_examples("url")) > 0

    drift_results = detect_drift(summary, prev_summaries)

    assert drift_results.is_drifted

    drifted_examples = drift_results.drifted_examples("url")

    assert len(drifted_examples["drifted_examples"]) > 0
    assert len(drifted_examples["corresponding_examples"]) > 0


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
)
def test_clustering_with_drift():
    num_rows = 100
    num_cols = 50
    data = np.random.rand(num_rows, num_cols)
    cols = [f"col_{i+1}" for i in range(num_cols)]
    date_range = pd.date_range(start="2022-01-01", periods=10, freq="D")

    df = pd.DataFrame(data, columns=cols)
    df["embedding_key"] = [
        "embedding_key_{}".format(i) for i in range(num_rows)
    ]
    df["embedding"] = [np.random.rand(1024) for _ in range(num_rows)]
    df["embedding_key2"] = [
        "embedding_key2_{}".format(i) for i in range(num_rows)
    ]
    df["embedding2"] = [np.random.rand(1024) + 1 for _ in range(num_rows)]
    df["date"] = np.random.choice(date_range[:-1], size=num_rows)

    prev_summaries = summarize(
        df,
        partition_key="date",
        columns=cols,
        embedding_column_map={
            "embedding_key": "embedding",
            "embedding_key2": "embedding2",
        },
    )

    drifted_df = pd.DataFrame(data[:50, :], columns=cols)
    drifted_df["date"] = [date_range[-1] for _ in range(50)]
    drifted_df["embedding"] = [np.random.rand(1024) * 10 for _ in range(50)]
    drifted_df["embedding2"] = [
        np.random.rand(1024) * 10 + 1 for _ in range(50)
    ]
    drifted_df["embedding_key"] = [
        "embedding_key_{}".format(i) for i in range(50)
    ]
    drifted_df["embedding_key2"] = [
        "embedding_key2_{}".format(i) for i in range(50)
    ]

    summary = summarize(drifted_df, previous_summaries=prev_summaries)

    drift_results = detect_drift(summary[0], prev_summaries)

    assert drift_results.is_drifted
    assert (
        "embedding2" in drift_results.drifted_columns().head(2).index.tolist()
    )
    assert (
        "embedding" in drift_results.drifted_columns().head(2).index.tolist()
    )
