import typing

import numpy as np
import pandas as pd
import polars as pl

from gate.statistics import compute_embeddings_examples, compute_embeddings_summary


class Summary:
    def __init__(
        self,
        summary: pd.DataFrame,
        embeddings_summary: pd.DataFrame,
        string_columns: typing.List[str],
        float_columns: typing.List[str],
        int_columns: typing.List[str],
        embedding_column_map: typing.Dict[str, str],
        embedding_examples: typing.Dict[str, pd.DataFrame],
        embedding_centroids: typing.Dict[str, np.ndarray],
        partition_key: str,
        partition_value: typing.Any,
    ):
        self._summary = summary
        self._embeddings_summary = embeddings_summary
        self._string_columns = string_columns
        self._float_columns = float_columns
        self._int_columns = int_columns
        self._embedding_columns = list(embedding_column_map.values())
        self._embedding_column_map = embedding_column_map
        self._embedding_examples = embedding_examples
        self._embedding_centroids = embedding_centroids
        self._partition_key = partition_key
        self._partition = partition_value

        if len(embedding_column_map) > 0:
            assert embedding_column_map.keys() == embedding_examples.keys()

    @property
    def summary(self) -> pd.DataFrame:
        """Dataframe containing the summary statistics."""
        return self._summary

    @property
    def embeddings_summary(self) -> pd.DataFrame:
        """Dataframe containing the embeddings summary statistics."""
        return self._embeddings_summary

    @property
    def partition(self) -> str:
        """Partition key value."""
        return self._partition

    @property
    def columns(self) -> typing.List[str]:
        """Columns for which summary statistics were computed."""
        return (
            self._string_columns
            + self._float_columns
            + self._int_columns
            + self._embedding_columns
        )

    @property
    def non_embedding_columns(self) -> typing.List[str]:
        """Columns for which summary statistics were computed.
        Ignores embedding columns."""
        return self._string_columns + self._float_columns + self._int_columns

    @property
    def partition_key(self) -> str:
        """Partition key column."""
        return self._partition_key

    def embedding_examples(self, embedding_key_column: str) -> pd.DataFrame:
        """Returns examples in each embedding cluster for the given
        embedding key column.

        Args:
            embedding_key_column (str):
                Column name representing the embedding key.

        Raises:
            ValueError: If there are no embedding examples.
            ValueError: If the embedding key column does not exist.

        Returns:
            pd.DataFrame: Examples in each embedding cluster. Contains
                the columns partition_key, embedding_key_column,
                embedding_value_column, and cluster.
        """
        if self._embedding_examples is None:
            raise ValueError("There are no embedding examples.")

        if embedding_key_column not in self._embedding_examples:
            raise ValueError(
                f"Embedding key column {embedding_key_column} does not exist."
                f" Valid columns are {self._embedding_examples.keys()}."
            )

        return self._embedding_examples[embedding_key_column]

    def embedding_centroids(self, embedding_key_column: str) -> np.ndarray:
        """Returns embedding centroids for the given embedding key column.

        Args:
            embedding_key_column (str):
                Column name representing the embedding key.

        Raises:
            ValueError: If there are no embedding examples.
            ValueError: If the embedding key column does not exist.

        Returns:
            np.ndarray:
                Matrix of embedding centroids, size (num_clusters, embedding_dim).
        """
        if self._embedding_centroids is None:
            raise ValueError("There are no embedding centroids.")

        if embedding_key_column not in self._embedding_centroids:
            raise ValueError(
                f"Embedding key column {embedding_key_column} does not exist."
                f" Valid columns are {self._embedding_centroids.keys()}."
            )

        return self._embedding_centroids[embedding_key_column]

    def statistics(self) -> typing.List[str]:
        """
        Returns list of statistics computed for each column:

        * coverage: Fraction of rows that are not null.
        * mean: Mean of the column.
        * p50: Median of the column.
        * num_unique_values: Number of unique values in the column.
        * occurrence_ratio: Ratio of the most common value to all other
        values.
        * p95: 95th percentile of the column.
        """
        value = self.value()
        statistics = value.columns.tolist()
        statistics.remove(self.partition_key)
        statistics.remove("column")
        return statistics

    @classmethod
    def fromRaw(
        cls,
        raw_data: pd.DataFrame,
        columns: typing.List[str] = [],
        embedding_column_map: typing.Dict[str, str] = {},
        partition_key: str = "",
        previous_summaries: typing.List["Summary"] = [],
    ) -> typing.List["Summary"]:
        polars_df = pl.DataFrame(raw_data)

        if len(previous_summaries) > 0:
            partition_key = previous_summaries[0].partition_key
            columns = previous_summaries[0].columns
            string_columns = previous_summaries[0]._string_columns
            float_columns = previous_summaries[0]._float_columns
            int_columns = previous_summaries[0]._int_columns
            embedding_column_map = previous_summaries[0]._embedding_column_map
        else:
            # Set up columns if it's the first partition
            assert (
                len(columns) > 0 or len(embedding_column_map) > 0
            ), "Must specify columns or embedding_column_map."

            if not set(columns).issubset(set(raw_data.columns)):
                raise ValueError(
                    "Columns to compute summaries on are not all in the dataframe."
                )
            types = polars_df.schema

            column_types = {c: types[c] for c in columns}
            string_columns = [c for c, t in column_types.items() if t == pl.Utf8]
            float_columns = [
                c for c, t in column_types.items() if t == pl.Float32 or t == pl.Float64
            ]
            int_columns = [
                c
                for c, t in column_types.items()
                if t == pl.Int64 or t == pl.Int32 or t == pl.Int16 or t == pl.Int8
            ]
            bool_columns = [c for c, t in column_types.items() if t == pl.Boolean]
            for c in bool_columns:
                polars_df = polars_df.with_columns([pl.col(c).cast(pl.Int8).alias(c)])
            int_columns += bool_columns

            assert len(string_columns) + len(float_columns) + len(int_columns) == len(
                columns
            ), "Columns have unknown type. Must be one of int, float, string,"

        if partition_key not in polars_df.columns:
            raise ValueError(
                f"Partition column {partition_key} is not in dataframe columns."
            )
        if not set(columns).issubset(set(polars_df.columns)):
            raise ValueError(
                "Columns to compute summaries on are not all in the dataframe."
            )

        # Compute the summary statistics
        statistics = [
            polars_df.groupby(partition_key)
            .agg(
                [
                    pl.col(c).is_not_null().mean().alias(c).cast(pl.Float32)
                    for c in string_columns + float_columns + int_columns
                ]
            )
            .with_columns([pl.lit("coverage").alias("statistic")]),
            polars_df.groupby(partition_key)
            .agg(
                [
                    pl.col(c).cast(pl.Float32).mean().alias(c)
                    for c in float_columns + int_columns
                ]
            )
            .with_columns([pl.lit("mean").alias("statistic")]),
            polars_df.groupby(partition_key)
            .agg(
                [
                    pl.col(c).quantile(0.5).cast(pl.Float32).alias(c)
                    for c in float_columns + int_columns
                ]
            )
            .with_columns([pl.lit("p50").alias("statistic")]),
            polars_df.groupby(partition_key)
            .agg(
                [
                    pl.col(c).approx_unique().cast(pl.Float32).alias(c)
                    for c in string_columns + int_columns
                ]
            )
            .with_columns([pl.lit("num_unique_values").alias("statistic")]),
            polars_df.groupby(partition_key)
            .agg(
                [
                    ((pl.col(c).unique_counts().max()) / (pl.col(c).count()))
                    .alias(c)
                    .cast(pl.Float32)
                    for c in string_columns + int_columns
                ]
            )
            .with_columns([pl.lit("occurrence_ratio").alias("statistic")]),
            polars_df.groupby(partition_key)
            .agg(
                [
                    pl.col(c).quantile(0.95).cast(pl.Float32).alias(c)
                    for c in float_columns + int_columns
                ]
            )
            .with_columns([pl.lit("p95").alias("statistic")]),
        ]
        statistics = pl.concat(statistics, how="diagonal").to_pandas()

        # Pivot such that columns are the statistics and rows are the row name,
        # and it's grouped by partition col

        pivoted_statistics = (
            statistics.melt(
                id_vars=[partition_key, "statistic"],
                value_vars=string_columns + float_columns + int_columns,
                var_name="column",
            )
            .pivot(
                index=[partition_key, "column"],
                columns="statistic",
                values="value",
            )
            .reset_index()
        )
        pivoted_statistics.columns = pivoted_statistics.columns.tolist()

        if len(embedding_column_map) > 0:
            # Embedding statistics
            full_embedding_df = compute_embeddings_summary(
                polars_df, embedding_column_map, partition_key
            )
            (
                embedding_example_map,
                embedding_centroids_map,
            ) = compute_embeddings_examples(
                polars_df,
                embedding_column_map,
                partition_key,
                num_clusters=5,
                num_examples=10,
            )

            embedding_statistics = [
                full_embedding_df.groupby(partition_key)
                .agg(
                    [
                        pl.col(c).is_not_null().mean().alias(c).cast(pl.Float32)
                        for c in full_embedding_df.columns[1:]
                    ]
                )
                .with_columns([pl.lit("coverage").alias("statistic")]),
                full_embedding_df.groupby(partition_key)
                .agg(
                    [
                        pl.col(c).cast(pl.Float32).mean().alias(c)
                        for c in full_embedding_df.columns[1:]
                    ]
                )
                .with_columns([pl.lit("mean").alias("statistic")]),
                full_embedding_df.groupby(partition_key)
                .agg(
                    [
                        pl.col(c).quantile(0.5).cast(pl.Float32).alias(c)
                        for c in full_embedding_df.columns[1:]
                    ]
                )
                .with_columns([pl.lit("p50").alias("statistic")]),
                full_embedding_df.groupby(partition_key)
                .agg(
                    [
                        pl.col(c).quantile(0.95).cast(pl.Float32).alias(c)
                        for c in full_embedding_df.columns[1:]
                    ]
                )
                .with_columns([pl.lit("p95").alias("statistic")]),
            ]

            embedding_statistics = pl.concat(
                embedding_statistics, how="diagonal"
            ).to_pandas()
            embedding_col_index = embedding_statistics.columns.tolist()
            embedding_col_index.remove("statistic")
            embedding_col_index.remove(partition_key)

            pivoted_embeddings = (
                embedding_statistics.melt(
                    id_vars=[partition_key, "statistic"],
                    value_vars=embedding_col_index,
                    var_name="column",
                )
                .pivot(
                    index=[partition_key, "column"],
                    columns="statistic",
                    values="value",
                )
                .reset_index()
            )
            pivoted_embeddings.columns = pivoted_embeddings.columns.tolist()

        groups = []
        for partition_value, group in pivoted_statistics.groupby(partition_key):
            relevant_embeddings = (
                pivoted_embeddings[
                    pivoted_embeddings[partition_key] == partition_value
                ].reset_index(drop=True)
                if len(embedding_column_map) > 0
                else None
            )

            groups.append(
                cls(
                    group.reset_index(drop=True),
                    relevant_embeddings,
                    string_columns,
                    float_columns,
                    int_columns,
                    embedding_column_map,
                    (
                        embedding_example_map[partition_value]
                        if len(embedding_column_map) > 0
                        else None
                    ),
                    (
                        embedding_centroids_map[partition_value]
                        if len(embedding_column_map) > 0
                        else None
                    ),
                    partition_key,
                    partition_value,
                )
            )

        # Handle if there are only embeddings columns
        if len(string_columns) + len(float_columns) + len(int_columns) == 0:
            for partition_value, group in pivoted_embeddings.groupby(partition_key):
                groups.append(
                    cls(
                        None,
                        group.reset_index(drop=True),
                        string_columns,
                        float_columns,
                        int_columns,
                        embedding_column_map,
                        embedding_example_map[partition_value],
                        embedding_centroids_map[partition_value],
                        partition_key,
                        partition_value,
                    )
                )

        return groups

    def value(self) -> pd.DataFrame:
        """Combines the summary and embeddings summary into a single dataframe.

        Returns:
            pd.DataFrame: Summary including embeddings, if exists.
        """
        if self.embeddings_summary is None:
            return self.summary

        if self.summary is None:
            return self.embeddings_summary

        return pd.concat([self.summary, self.embeddings_summary], ignore_index=True)

    def __str__(self) -> str:
        """
        String representation of the object's value (i.e., summary).

        Usage: `print(summary)`
        """

        if self.embeddings_summary is None:
            return f"Summary:\n{self.summary.to_string()}"

        if self.summary is None:
            return f"Embedding summary:\n{self.embeddings_summary.to_string()}"

        return (
            f"Regular summary:\n{self.summary.to_string()}\nEmbedding"
            f" summary:\n{self.embeddings_summary.to_string()}"
        )
