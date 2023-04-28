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
        summary: Summary,
        neighbor_summaries: typing.List[Summary],
        clustered_features: pd.DataFrame,
        embedding_columns: typing.List[str],
    ) -> None:
        self._all_scores = all_scores
        self._nn_features = nn_features
        self._summary = summary
        self._neighbor_summaries = neighbor_summaries
        self._clustered_features = clustered_features
        self._embedding_columns = embedding_columns

    @property
    def summary(self) -> Summary:
        """Summary of the partition."""
        return self._summary

    @property
    def neighbor_summaries(self) -> typing.List[Summary]:
        """Summaries of the nearest neighbors of the partition."""
        return self._neighbor_summaries

    def drifted_examples(
        self, embedding_key_column: str
    ) -> typing.Dict[str, pd.DataFrame]:
        """Returns some examples from the partition that are
        most drifted from nearest neighbors in the embedding space
        in previous partitions.

        Throws an error if the embedding_key_column isn't a valid
        embedding key column, or if there are no embedding columns.

        Args:
            embedding_key_column (str):
                Column that represents the embedding key (e.g., text, image).

        Returns:
            typing.Dict[str, pd.DataFrame]:
                Dictionary with two keys: "drifted_examples" and
                "corresponding_examples". The value of each key is a
                dataframe with columns "partition_key", "embedding_key_column",
                and "embedding_value_column".
        """
        all_centroids = np.vstack(
            [
                s.embedding_centroids(embedding_key_column)
                for s in self.neighbor_summaries
            ]
        )
        all_centroid_idxs = [
            (i, j)
            for i in range(len(self.neighbor_summaries))
            for j in range(
                len(
                    self.neighbor_summaries[i].embedding_centroids(embedding_key_column)
                )
            )
        ]
        curr_centroids = self.summary.embedding_centroids(embedding_key_column)

        # Compute similarity
        similarity_matrix = cosine_similarity(curr_centroids, all_centroids)
        most_dissimilar_row_idx = np.argmax(np.min(similarity_matrix, axis=1))
        dissimilar_examples = self.summary.embedding_examples(embedding_key_column)
        dissimilar_examples = dissimilar_examples[
            dissimilar_examples["cluster"] == most_dissimilar_row_idx
        ].reset_index(drop=True)
        corresponding_row_idx = np.argmin(similarity_matrix[most_dissimilar_row_idx])
        corresponding_examples = self.neighbor_summaries[
            all_centroid_idxs[corresponding_row_idx][0]
        ].embedding_examples(embedding_key_column)
        corresponding_examples = corresponding_examples[
            corresponding_examples["cluster"]
            == all_centroid_idxs[corresponding_row_idx][1]
        ].reset_index(drop=True)

        return {
            "drifted_examples": dissimilar_examples.drop("cluster", axis=1),
            "corresponding_examples": corresponding_examples.drop("cluster", axis=1),
        }

    @property
    def score(self) -> float:
        """Distance from the partition to its k nearest neighbors."""
        return self._all_scores[self.summary.partition]

    @property
    def is_drifted(self) -> bool:
        """
        Indicates whether the partition is drifted or not, compared
        to previous partitions. This is determined by the percentile
        of the partition's score in the distribution of all scores.
        The threshold is 95%.
        """
        return self.score_percentile >= 0.95

    @property
    def score_percentile(self) -> float:
        """Percentile of the partition's score in the distribution
        of all scores."""
        return percentileofscore(self.all_scores, self.score) * 1.0 / 100

    @property
    def all_scores(self) -> pd.Series:
        """Scores of all previous partitions."""
        mask = self._all_scores.index != self.summary.partition
        return self._all_scores[mask]

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

    def drill_down(
        self,
        sort_by_cluster_score: bool = False,
        average_embedding_columns: bool = True,
    ) -> pd.DataFrame:
        """Compute the columns with highest magnitude anomaly scores.
        Anomaly scores are computed as the z-score of the column with
        respect to previous partition summary statistics.

        The resulting dataframe has the following schema (column, statistic are
        indexes):

        - column: Name of the column
        - statistic: Name of the statistic
        - z-score: z-score of the column
        - cluster: Cluster number that the column belongs to (if clustering was
        performed)
        - abs(z-score-cluster): absolute value of the average z-score of the
        column in the cluster (if clustering was performed)

        Use the `drifted_columns` method first, since `drifted_columns`
        deduplicates columns.

        Args:
            sort_by_cluster_score (bool, optional):
                Whether to sort by cluster z-score. Defaults to False.
            average_embedding_columns (bool, optional):
                Whether to average statistics across embedding dimensions.
                Defaults to True.

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
        last_day = self._nn_features.loc[self.summary.partition]
        sorted_cols = last_day.abs().sort_values(ascending=False).index
        sorted_df = last_day[sorted_cols].to_frame()
        sorted_df.rename(columns={sorted_df.columns[0]: "z-score"}, inplace=True)
        sorted_df = sorted_df.rename_axis(["column", "statistic"])

        if self._clustered_features is not None:
            # Join the clustered features with the sorted_df
            # sorted_df.rename(index={"column": "cluster"}, inplace=True)
            sorted_df = sorted_df.rename_axis(["cluster", "statistic"]).reset_index()
            sorted_df.rename(columns={"z-score": "abs(z-score-cluster)"}, inplace=True)

            sorted_df = sorted_df.merge(
                self._clustered_features,
                on=["cluster", "statistic"],
                how="left",
            )

            # Sort again
            if sort_by_cluster_score:
                sorted_df = sorted_df.reindex(
                    sorted_df[["abs(z-score-cluster)", "z-score"]]
                    .abs()
                    .sort_values(
                        by=["abs(z-score-cluster)", "z-score"], ascending=False
                    )
                    .index
                )
                sorted_df.set_index(["column", "statistic"], inplace=True)

        if len(self._embedding_columns) > 0 and average_embedding_columns:
            # Average the z-scores
            sorted_df.reset_index(inplace=True)
            sorted_df["column"] = sorted_df["column"].apply(
                lambda x: name_to_ec(x, self._embedding_columns)
            )
            sorted_df["z-score"] = sorted_df.apply(
                lambda x: (
                    abs(x["z-score"])
                    if x["column"] in self._embedding_columns
                    else x["z-score"]
                ),
                axis=1,
            )
            sorted_df = sorted_df.groupby(["column", "statistic"]).mean()
            sorted_df = sorted_df.reindex(
                sorted_df["z-score"].abs().sort_values(ascending=False).index
            )

            # sorted_df.sort_values(by="z-score", ascending=False, inplace=True)

        return sorted_df

    def __str__(self) -> str:
        """Prints the drift score, percentile, and the top drifted columns."""
        results = (
            "Drift score:"
            f" {self.score:.4f} ({self.score_percentile:.2%} percentile)\nTop"
            f" drifted columns:\n{self.drifted_columns()}"
        )
        return results

    def drifted_columns(
        self,
        limit: int = 10,
        average_embedding_columns: bool = True,
    ) -> pd.DataFrame:
        """Returns the top limit columns that have drifted. The
        resulting dataframe has the following schema (column is an
        index):

        - column: Name of the column
        - statistic: Name of the statistic
        - z-score: z-score of the column
        - cluster: Cluster number of the column (if clustering was performed)
        - abs(z-score-cluster): z-score of the column in the cluster (if
        clustering was performed)

        Args:
            limit (int, optional):
                Limit for number of drifted columns to return. Defaults to 10.
            average_embedding_columns (bool, optional):
                Whether to average statistics across embedding dimensions.
                Defaults to True.

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
        dd_results = self.drill_down(average_embedding_columns)

        if self._clustered_features is not None:
            # Sort by z-score first, then abs(z-score-cluster)
            dd_results = dd_results.reindex(
                dd_results[["z-score", "abs(z-score-cluster)"]]
                .abs()
                .sort_values(by=["z-score", "abs(z-score-cluster)"], ascending=False)
                .index
            )

        dd_results.reset_index(inplace=True)

        dd_results.drop_duplicates(subset=["column"], keep="first", inplace=True)
        dd_results.set_index("column", inplace=True)

        if self._clustered_features is not None:
            # Reorder columns
            dd_results = dd_results[
                ["statistic", "z-score", "cluster", "abs(z-score-cluster)"]
            ]
            dd_results = dd_results[dd_results["abs(z-score-cluster)"].abs() > 0.0]

        return dd_results.head(limit)


def name_to_ec(name: str, embedding_columns: typing.List[str]) -> str:
    """Converts a column name to an embedding column name.

    Args:
        name (str):
            Column name.
        embedding_columns (typing.List[str]):
            List of embedding columns.

    Returns:
        str:
            Embedding column name.
    """
    if type(name) != str:
        print(name)
    split_name = name.rsplit("_", 1)[0]
    if split_name in embedding_columns:
        return split_name
    else:
        return name


def detect_drift(
    current_summary: Summary,
    previous_summaries: typing.List[Summary],
    validity: typing.List[int] = [],
    cluster: bool = True,
    k: int = 3,
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
            Defaults to 3.

    Returns (DriftResult): DriftResult object with score and score percentile.
    """
    if len(previous_summaries) < 5:
        raise ValueError(
            "You must have at least 5 previous partition summary to detect"
            " drift. You can randomly split your data from previous partitions"
            " into 5+ partitions if you need to."
        )

    partition_key = current_summary.partition_key
    columns = current_summary.columns
    statistics = current_summary.statistics()

    # Create validity vector
    if not validity:
        validity = [1] * len(previous_summaries)
    if len(validity) != len(previous_summaries):
        raise ValueError(
            f"Validity vector has length {len(validity)} but should have"
            f" length {len(previous_summaries)} to match previous_summaries."
        )
    validity.append(1)

    # Normalize current and previous partition summaries
    prev_summaries = [
        s.value() for i, s in enumerate(previous_summaries) if validity[i] == 1
    ]
    normalized_summaries = normalize(
        pd.concat(prev_summaries + [current_summary.value()]).reset_index(drop=True),
        partition_key,
        statistics,
    )

    # Run clustering algorithm if there are more than 10 columns
    if cluster and len(columns) >= 10:
        clustering = compute_clusters(
            normalized_summaries,
            partition_key,
            current_summary._string_columns,
            current_summary._float_columns,
            current_summary._int_columns,
            current_summary._embedding_columns,
        )

        normalized_summaries["value_abs"] = normalized_summaries["value"].abs()

        cluster_normalized = (
            normalized_summaries.merge(clustering, on=["column"], how="left")
            # .set_index(partition_key)
            .groupby([partition_key, "cluster", "statistic"])["value_abs"]
            .mean()
            .reset_index()
        )
        cluster_normalized.rename(
            {"cluster": "column", "value_abs": "value"}, axis=1, inplace=True
        )
        normalized_summaries.drop("value_abs", axis=1, inplace=True)

    # Run nearest neighbor algorithm to get distances
    nn_features_unpivoted = (
        cluster_normalized if (cluster and len(columns) >= 10) else normalized_summaries
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

    dists, indices = cKDTree(nn_features.values).query(nn_features.values, k=k + 1)

    neighbor_partitions = nn_features.index[indices[-1][1:]].to_list()
    neighbor_summaries = [
        s for s in previous_summaries if s.partition in neighbor_partitions
    ]

    # Replace inf with nan
    dists[np.isinf(dists)] = np.nan

    scores = pd.Series(
        data=np.nanmean(dists[:, 1:], axis=1),
        index=nn_features.index,
    )

    if cluster and len(columns) >= 10:
        partition_value = scores.index[-1]
        clustered_features = normalized_summaries[
            normalized_summaries[partition_key] == partition_value
        ].merge(clustering, on=["column"], how="left")

        clustered_features.rename({"value": "z-score"}, axis=1, inplace=True)
        clustered_features.drop(partition_key, axis=1, inplace=True)

        return DriftResult(
            scores,
            nn_features,
            current_summary,
            neighbor_summaries=neighbor_summaries,
            clustered_features=clustered_features,
            embedding_columns=current_summary._embedding_columns,
        )

    else:
        return DriftResult(
            scores,
            nn_features,
            current_summary,
            neighbor_summaries=neighbor_summaries,
            clustered_features=None,
            embedding_columns=current_summary._embedding_columns,
        )


def normalize(
    all_summaries: pd.DataFrame,
    partition_key: str,
    statistics: typing.List[str],
) -> pd.DataFrame:
    """Melt and normalize partition summaries.

    Args:
        all_summaries (pd.DataFrame): concatenated summaries to normalize
        partition_key (str): partition key
        statistics (typing.List[str]): statistics to normalize

    Returns:
        pd.DataFrame: normalized summary
    """
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
    return normalized


def compute_clusters(
    normalized: pd.DataFrame,
    partition_key: str,
    string_columns: typing.List[str],
    float_columns: typing.List[str],
    int_columns: typing.List[str],
    embedding_columns: typing.List[str],
) -> pd.DataFrame:
    """Computes clusters of columns in a partition summary.

    Args:
        normalized (pd.DataFrame): Normalized partition summary.
        partition_key (str): Name of partition column.
        string_columns (typing.List[str]): List of string columns.
        float_columns (typing.List[str]): List of float columns.
        int_columns (typing.List[str]): List of int columns.
        embedding_columns (typing.List[str]): List of embedding columns.

    Returns (pd.DataFrame): Mapping of column names to cluster numbers.
    """

    column_stats = normalized.pivot_table(
        index="column", columns=[partition_key, "statistic"], values="value"
    ).fillna(0.0)

    column_names = column_stats.index.tolist()
    column_names_to_types = {c: "string" for c in string_columns}
    column_names_to_types.update({c: "float" for c in float_columns})
    column_names_to_types.update({c: "int" for c in int_columns})
    embedding_columns_with_indexes = [
        c for c in column_names if name_to_ec(c, embedding_columns) in embedding_columns
    ]
    # column_names_to_types.update(
    #     {c: "embedding" for c in embedding_columns_with_indexes}
    # )
    for embedding_col_name in embedding_columns_with_indexes:
        column_names.remove(embedding_col_name)

    model = SentenceTransformer("clip-ViT-B-32")
    embeddings = model.encode(
        [f"{c} is of type {column_names_to_types[c]}" for c in column_names]
    )

    embedding_similarity_matrix = cosine_similarity(embeddings)
    value_similarity_matrix = cosine_similarity(
        column_stats[column_stats.index.isin(column_names)].values
    )
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
    max_label = max(cluster_labels)

    # Add embedding columns to clusters
    for i, embedding_col_name in enumerate(embedding_columns):
        for name in column_stats.index.tolist():
            if name_to_ec(name, embedding_columns) == embedding_col_name:
                clusters[name] = max_label + i + 1

    cluster_df = (
        pd.DataFrame.from_dict(clusters, orient="index", columns=["cluster"])
        .reset_index()
        .rename(columns={"index": "column"})
    )
    # cluster_df["cluster"] = cluster_df["cluster"].astype(str)

    return cluster_df
