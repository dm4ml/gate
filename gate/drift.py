import typing

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import percentileofscore
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from gate.summary import Summary


class DriftResult:
    def __init__(
        self,
        all_scores: pd.Series,
        nn_features: pd.DataFrame,
        clustered_features: pd.DataFrame,
    ) -> None:
        self._all_scores = all_scores
        self._nn_features = nn_features
        self._clustered_features = clustered_features

    @property
    def score(self) -> float:
        """Distance from the partition to its k nearest neighbors."""
        return self._all_scores.iloc[-1]

    @property
    def is_drifted(self) -> bool:
        """
        Indicates whether the partition is drifted or not, compared
        to previous partitions. This is determined by the percentile
        of the partition's score in the distribution of all scores.
        The threshold is 90%.
        """
        return self.score_percentile >= 0.85

    @property
    def score_percentile(self) -> float:
        """Percentile of the partition's score in the distribution
        of all scores."""
        return percentileofscore(self.all_scores, self.score) * 1.0 / 100

    @property
    def all_scores(self) -> pd.Series:
        """Scores of all previous partitions."""
        return self._all_scores.iloc[:-1]

    @property
    def clustering(self) -> typing.Dict[int, typing.List[str]]:
        """
        Clustering of the columns based on their partition summaries
        and meaning of column names (determined via embeddings). Returns
        a dictionary with cluster numbers as keys and lists of columns
        as values.
        """
        if self._clustered_features is None:
            raise ValueError("No clustering was performed.")

        clustering_map = self._clustered_features.groupby("cluster")["column"].agg(set)
        clustering_map = clustering_map.apply(list)

        return clustering_map.to_dict()

    def drill_down(self) -> pd.DataFrame:
        """Compute the columns with highest magnitude anomaly scores.
        Anomaly scores are computed as the z-score of the column with
        respect to previous partition summary statistics.

        The resulting dataframe has the following schema (column, statistic are
        indexes):

        - column: Name of the column
        - statistic: Name of the statistic
        - z-score: z-score of the column
        - cluster: Cluster number of the column (if clustering was performed)
        - z-score-cluster: z-score of the column in the cluster (if
        clustering was performed)

        Use the `drifted_columns` method first, since `drifted_columns`
        deduplicates columns.

        Returns:
            pd.DataFrame:
                Dataframe with columns with highest magnitude anomaly
                scores. Sorted by the magnitude of the z-score for a column.
                If clustering was performed, the dataframe will be sorted
                by the magnitude of the z-score in the cluster before
                the column score.
        """

        # Return a dataframe with features with highest magnitude anomaly
        # scores

        last_day = self._nn_features.iloc[-1]
        sorted_cols = last_day.abs().sort_values(ascending=False).index
        sorted_df = last_day[sorted_cols].to_frame()
        sorted_df.rename(columns={sorted_df.columns[0]: "z-score"}, inplace=True)
        sorted_df = sorted_df.rename_axis(["column", "statistic"])

        if self._clustered_features is not None:
            # Join the clustered features with the sorted_df
            # sorted_df.rename(index={"column": "cluster"}, inplace=True)
            sorted_df = sorted_df.rename_axis(["cluster", "statistic"]).reset_index()
            sorted_df.rename(columns={"z-score": "z-score-cluster"}, inplace=True)

            sorted_df = sorted_df.merge(
                self._clustered_features,
                on=["cluster", "statistic"],
                how="left",
            )

            # Sort again
            sorted_df = sorted_df.reindex(
                sorted_df[["z-score-cluster", "z-score"]]
                .abs()
                .sort_values(by=["z-score-cluster", "z-score"], ascending=False)
                .index
            )
            sorted_df.set_index(["column", "statistic"], inplace=True)

        return sorted_df

    def __str__(self) -> str:
        """Prints the drift score, percentile, and the top drifted columns."""
        results = (
            "Drift score:"
            f" {self.score:.4f} ({self.score_percentile:.2%} percentile)\nTop"
            f" drifted columns:\n{self.drifted_columns()}"
        )
        return results

    def drifted_columns(self, limit: int = 10) -> pd.DataFrame:
        """Returns the top limit columns that have drifted. The
        resulting dataframe has the following schema (column is an
        index):

        - column: Name of the column
        - statistic: Name of the statistic
        - z-score: z-score of the column
        - cluster: Cluster number of the column (if clustering was performed)
        - z-score-cluster: z-score of the column in the cluster (if
        clustering was performed)

        Args:
            limit (int, optional):
                Limit for number of drifted columns to return. Defaults to 10.

        Returns:
            pd.DataFrame:
                Dataframe with columns with highest magnitude z-scores.
                If clustering was performed, the dataframe will also contain
                the z-score in the cluster and the cluster number.
                Each column is deduplicated, so only the statistic with the
                highest magnitude z-score is returned.
        """
        # Return a dataframe of the top limit columns that have drifted
        # Drop duplicate column names
        dd_results = self.drill_down()

        if self._clustered_features is not None:
            # Sort by z-score first, then z-score-cluster
            dd_results = dd_results.reindex(
                dd_results[["z-score-cluster", "z-score"]]
                .abs()
                .sort_values(by=["z-score", "z-score-cluster"], ascending=False)
                .index
            )

        dd_results.reset_index(inplace=True)

        dd_results.drop_duplicates(subset=["column"], keep="first", inplace=True)
        dd_results.set_index("column", inplace=True)

        if self._clustered_features is not None:
            # Reorder columns
            dd_results = dd_results[
                ["statistic", "z-score", "cluster", "z-score-cluster"]
            ]
            dd_results = dd_results[dd_results["z-score-cluster"].abs() > 0.0]

        return dd_results.head(limit)


def detect_drift(
    current_summary: Summary,
    previous_summaries: typing.List[Summary],
    validity: typing.List[int] = [],
    cluster: bool = True,
    k: int = 5,
) -> DriftResult:
    """Computes whether the current partition summary has drifted from previous
    summaries.

    Args:
        current_summary (Summary):
            Partition summary for current partition.
        previous_summaries (typing.List[Summary]):
            Previous partition summaries.
        validity (typing.List[int], optional):
            Indicator list identifying which partition summaries are valid. 1
            if valid, 0 if invalid. If empty, we assume all partition summaries
            are valid. Must be empty or equal to length of previous_summaries.
        cluster (bool, optional):
            Whether or not to cluster columns in summaries. Increases runtime
            but also increases precision in drift detection. Only engaged if
            summaries have more than 10 columns. Defaults to True.
        k (int, optional):
            Number of nearest neighbor partitions to inspect.
            Defaults to 5.

    Returns (DriftResult): DriftResult object with score and score percentile.
    """
    if len(previous_summaries) == 0:
        raise ValueError(
            "You must have at least 1 previous partition summary to detect drift."
        )

    partition_key = current_summary.partition_key
    columns = current_summary.columns
    statistics = current_summary.statistics

    # Create validity vector
    if not validity:
        validity = [1] * len(previous_summaries)
    if len(validity) != len(previous_summaries):
        raise ValueError(
            f"Validity vector has length {len(validity)} but should have"
            f" length {len(previous_summaries)} to match previous_summaries."
        )
    validity.append(1)

    prev_summaries = [
        s.value for i, s in enumerate(previous_summaries) if validity[i] == 1
    ]

    # Normalize current and previous partition summaries
    all_summaries = pd.concat(prev_summaries + [current_summary.value]).reset_index(
        drop=True
    )

    normalized = all_summaries.melt(
        id_vars=[partition_key, "column"],
        value_vars=statistics,
        var_name="statistic",
        value_name="value",
    ).dropna()

    grouped = normalized.groupby(["column", "statistic"])
    mean = grouped["value"].transform("mean")
    std = grouped["value"].transform("std")
    std += 1e-10
    normalized["value"] = (normalized["value"] - mean) / std

    # Run clustering algorithm if there are more than 10 columns
    if cluster and len(columns) >= 10:
        clustering = compute_clusters(
            normalized,
            partition_key,
            current_summary._string_columns,
            current_summary._float_columns,
            current_summary._int_columns,
        )

        normalized["value_abs"] = normalized["value"].abs()

        cluster_normalized = (
            normalized.merge(clustering, on=["column"], how="left")
            # .set_index(partition_key)
            .groupby([partition_key, "cluster", "statistic"])["value_abs"]
            .mean()
            .reset_index()
        )
        cluster_normalized.rename(
            {"cluster": "column", "value_abs": "value"}, axis=1, inplace=True
        )
        normalized.drop("value_abs", axis=1, inplace=True)

    # Run nearest neighbor algorithm to get distances
    nn_features_unpivoted = (
        cluster_normalized if (cluster and len(columns) >= 10) else normalized
    )
    # nn_features_unpivoted["value"] = nn_features_unpivoted["value"].apply(
    #     lambda x: 0.0 if np.abs(x) < z_score_cutoff else x
    # )

    nn_features = (
        nn_features_unpivoted.fillna(0.0)
        .pivot_table(
            index=partition_key,
            columns=["column", "statistic"],
            values="value",
        )
        .fillna(0.0)
    )

    dists, _ = cKDTree(nn_features.values).query(nn_features.values, k=k + 1)

    # Replace inf with nan
    dists[np.isinf(dists)] = np.nan

    scores = pd.Series(
        data=np.nanmean(dists[:, 1:], axis=1),
        index=nn_features.index,
    )

    if cluster and len(columns) >= 10:
        partition_value = scores.index[-1]
        clustered_features = normalized[
            normalized[partition_key] == partition_value
        ].merge(clustering, on=["column"], how="left")

        clustered_features.rename({"value": "z-score"}, axis=1, inplace=True)
        clustered_features.drop(partition_key, axis=1, inplace=True)

        return DriftResult(
            scores,
            nn_features,
            clustered_features=clustered_features,
        )

    else:
        return DriftResult(scores, nn_features, clustered_features=None)


def compute_clusters(
    normalized: pd.DataFrame,
    partition_key: str,
    string_columns: typing.List[str],
    float_columns: typing.List[str],
    int_columns: typing.List[str],
) -> pd.DataFrame:
    """Computes clusters of columns in a partition summary.

    Args:
        normalized (pd.DataFrame): Normalized partition summary.
        partition_key (str): Name of partition column.
        string_columns (typing.List[str]): List of string columns.
        float_columns (typing.List[str]): List of float columns.
        int_columns (typing.List[str]): List of int columns.

    Returns (pd.DataFrame): Mapping of column names to cluster numbers.
    """

    column_stats = normalized.pivot_table(
        index="column", columns=[partition_key, "statistic"], values="value"
    ).fillna(0.0)

    column_names = column_stats.index.values
    column_names_to_types = {c: "string" for c in string_columns}
    column_names_to_types.update({c: "float" for c in float_columns})
    column_names_to_types.update({c: "int" for c in int_columns})

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        [f"{c} is of type {column_names_to_types[c]}" for c in column_names]
    )

    embedding_similarity_matrix = cosine_similarity(embeddings)
    value_similarity_matrix = cosine_similarity(column_stats.values)
    similarity_matrix = (
        0.25 * embedding_similarity_matrix + 0.75 * value_similarity_matrix
    )

    # Run PCA on similarity matrix to get number of clusters
    pca = PCA(random_state=42)
    pca.fit(similarity_matrix)
    cumev = np.cumsum(pca.explained_variance_ratio_)
    # Find cluster cutoff
    cutoff = -1
    PCA_THRESHOLD = 0.95
    for idx, elem in enumerate(cumev):
        if elem > PCA_THRESHOLD:
            cutoff = idx
            break

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        n_clusters=cutoff + 1,
    )
    clustering.fit(similarity_matrix)

    # Aggregate columns based on clustering labels

    cluster_labels = clustering.labels_
    clusters = {}
    for i, label in enumerate(cluster_labels):
        clusters[column_names[i]] = label

    return (
        pd.DataFrame.from_dict(clusters, orient="index", columns=["cluster"])
        .reset_index()
        .rename(columns={"index": "column"})
    )
