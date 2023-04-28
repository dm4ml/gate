import typing

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans


def type_to_statistics(t: str) -> typing.List[str]:
    """Returns the statistics that are computed for a given type.

    Args:
        t (str): Type (one of "int", "float", "string", "embedding").

    Returns:
        typing.List[str]:
            List of statistics that are computed for the type.
            Partition summaries will have NaNs for statistics that are not computed.

    Raises:
        ValueError: If the type is unknown.
    """

    if t == "int":
        return [
            "coverage",
            "mean",
            "p50",
            # "stdev",
            "num_unique_values",
            "occurrence_ratio",
            "p95",
        ]

    if t == "float":
        return [
            "coverage",
            "mean",
            "p50",
            # "stdev",
            "p95",
        ]

    if t == "string":
        return ["coverage", "num_unique_values", "occurrence_ratio"]

    if t == "embedding":
        return ["coverage", "mean", "p50", "p95"]

    raise ValueError(f"Unknown type {t}")


def cluster(
    group: pd.DataFrame,
    embedding_value_column: str,
    num_clusters: int,
    num_examples: int,
    limit: int = 2000,
):
    shuffled = group.sample(limit, random_state=42) if len(group) > limit else group
    matrix = np.vstack(shuffled[embedding_value_column].apply(np.array).values)

    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init="auto",
        random_state=42,
    )
    kmeans.fit(matrix)
    labels = kmeans.labels_
    shuffled["cluster"] = labels
    centroids = kmeans.cluster_centers_

    # Select examples from each cluster
    examples = (
        shuffled.groupby("cluster")
        .apply(lambda x: x.sample(num_examples) if len(x) > num_examples else x)
        .reset_index(drop=True)
    )

    return examples, centroids


def compute_embeddings_examples(
    polars_df: pl.DataFrame,
    embedding_column_map: typing.Dict[str, str],
    partition_key: str,
    num_clusters: int,
    num_examples: int,
) -> typing.Tuple[
    typing.Dict[str, typing.Dict[str, pd.DataFrame]],
    typing.Dict[str, typing.Dict[str, np.ndarray]],
]:
    """Computes examples and centroids to store in each partition
    summary for each embedding column.

    Args:
        polars_df (pl.DataFrame): DataFrame with the embeddings.
        embedding_column_map (typing.Dict[str, str]):
            Map from embedding key column to embedding value column.
        partition_key (str): Column to partition by.
        num_clusters (int): Number of clusters to use in KMeans to
            cluster the embeddings.
        num_examples (int): Number of examples from each cluster to store
            in the partition summary.

    Returns:
        typing.Tuple[typing.Dict[str, typing.Dict[str, pd.DataFrame]], typing.
        Dict[str, typing.Dict[str, np.ndarray]]]:
            Examples and centroids to store in each partition summary.
    """

    all_examples = {}
    all_centroids = {}
    for (
        embedding_key_column,
        embedding_value_column,
    ) in embedding_column_map.items():
        # Select examples
        for partition_value, group in (
            polars_df[[partition_key, embedding_key_column, embedding_value_column]]
            .to_pandas()
            .groupby(partition_key)
        ):
            examples, centroids = cluster(
                group, embedding_value_column, num_clusters, num_examples
            )

            if partition_value not in all_examples:
                all_examples[partition_value] = {}

            if partition_value not in all_centroids:
                all_centroids[partition_value] = {}

            all_examples[partition_value][embedding_key_column] = examples
            all_centroids[partition_value][embedding_key_column] = centroids

    return all_examples, all_centroids


def compute_embeddings_summary(
    polars_df: pl.DataFrame,
    embedding_column_map: typing.Dict[str, str],
    partition_key: str,
):
    embedding_dfs = []
    for (
        _,
        embedding_value_column,
    ) in embedding_column_map.items():
        lengths = polars_df.select(
            pl.col(embedding_value_column).arr.lengths().alias("lengths")
        )
        num_lengths = lengths["lengths"].n_unique()
        if num_lengths > 1:
            raise ValueError(
                f"Embedding value column {embedding_value_column} has"
                " different lengths. All embeddings must have the same"
                " length."
            )
        length = lengths["lengths"].head(1)[0]

        embedding_df = polars_df.select(
            [pl.col(partition_key)]
            + [
                pl.col(embedding_value_column)
                .arr.get(i)
                .alias(embedding_value_column + f"_{i}")
                for i in range(length)
            ]
        )
        embedding_dfs.append(embedding_df)

    full_embedding_df = embedding_dfs[0]
    for embedding_df in embedding_dfs[1:]:
        full_embedding_df = full_embedding_df.join(embedding_df, on=partition_key)

    return full_embedding_df
