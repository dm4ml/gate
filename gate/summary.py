import typing

import pandas as pd
import polars as pl


class Summary:
    def __init__(
        self,
        value: pd.DataFrame,
        string_columns: typing.List[str],
        float_columns: typing.List[str],
        int_columns: typing.List[str],
        partition_key: str,
    ):
        self._value = value
        self._string_columns = string_columns
        self._float_columns = float_columns
        self._int_columns = int_columns
        self._partition_key = partition_key
        self._partition = value[partition_key].iloc[0]

    @property
    def value(self) -> pd.DataFrame:
        """Dataframe containing the summary statistics."""
        return self._value

    @property
    def partition(self) -> str:
        """Partition key value."""
        return self._partition

    @property
    def columns(self) -> typing.List[str]:
        """Columns for which summary statistics were computed."""
        return self._string_columns + self._float_columns + self._int_columns

    @property
    def partition_key(self) -> str:
        """Partition key column."""
        return self._partition_key

    @staticmethod
    def statistics() -> typing.List[str]:
        """
        Returns list of statistics computed for each column:

        * coverage: Fraction of rows that are not null.
        * mean: Mean of the column.
        * stdev: Standard deviation of the column.
        * num_unique_values: Number of unique values in the column.
        * occurrence_ratio: Ratio of the most common value to all other
        values.
        * num_frequent_values: Number of values that occur more than once.
        """
        return [
            "coverage",
            "mean",
            "stdev",
            "num_unique_values",
            "occurrence_ratio",
            # "num_frequent_values",
            "p95",
        ]

    @classmethod
    def fromRaw(
        cls,
        raw_data: pd.DataFrame,
        columns: typing.List[str] = [],
        partition_key: str = "",
        previous_summaries: typing.List["Summary"] = [],
    ) -> typing.List["Summary"]:
        if len(previous_summaries) > 0:
            partition_key = previous_summaries[0].partition_key
            columns = previous_summaries[0].columns
            string_columns = previous_summaries[0]._string_columns
            float_columns = previous_summaries[0]._float_columns
            int_columns = previous_summaries[0]._int_columns
        else:
            # Set up columns if it's the first partition
            assert len(columns) > 0

            if not set(columns).issubset(set(raw_data.columns)):
                raise ValueError(
                    "Columns to compute summaries on are not all in the dataframe."
                )
            types = raw_data.dtypes.to_dict()
            column_types = {c: types[c] for c in columns}
            string_columns = [c for c, t in column_types.items() if t == "O"]
            float_columns = [c for c, t in column_types.items() if t == "float"]
            int_columns = [c for c, t in column_types.items() if t == "int"]
            assert len(string_columns) + len(float_columns) + len(int_columns) == len(
                columns
            )

        if partition_key not in raw_data.columns:
            raise ValueError(
                f"Partition column {partition_key} is not in dataframe columns."
            )
        if not set(columns).issubset(set(raw_data.columns)):
            raise ValueError(
                "Columns to compute summaries on are not all in the dataframe."
            )

        # Compute the summary statistics

        # Convert to polars and melt
        polars_df = pl.DataFrame(raw_data)
        polars_df[partition_key].n_unique()
        # .melt(
        #     id_vars=[partition_key],
        #     value_vars=columns,
        #     variable_name="column",
        #     value_name="value",
        # )

        statistics = {
            "coverage": polars_df.groupby(partition_key).agg(
                [pl.col(c).is_not_null().mean().alias(c) for c in columns]
            ),
            "mean": polars_df.groupby(partition_key).agg(
                [pl.col(c).mean().alias(c) for c in float_columns + int_columns]
            ),
            "stdev": polars_df.groupby(partition_key).agg(
                [
                    pl.col(c).std().cast(pl.Float64).alias(c)
                    for c in float_columns + int_columns
                ]
            ),
            "num_unique_values": polars_df.groupby(partition_key).agg(
                [
                    pl.col(c).approx_unique().cast(pl.Float64).alias(c)
                    for c in string_columns + int_columns
                ]
            ),
            "occurrence_ratio": polars_df.groupby(partition_key).agg(
                [
                    (pl.col(c).unique_counts().max())
                    / (pl.col(c).count()).cast(pl.Float64).alias(c)
                    for c in string_columns + int_columns
                ]
            ),
            "p95": polars_df.groupby(partition_key).agg(
                [
                    pl.col(c).quantile(0.95).cast(pl.Float64).alias(c)
                    for c in float_columns + int_columns
                ]
            ),
        }

        # Merge the statistics into a single dataframe
        for name, df in statistics.items():
            statistics[name] = df.with_columns([pl.lit(name).alias("statistic")])

        current_statistics = pl.concat(
            list(statistics.values()), how="diagonal"
        ).to_pandas()

        # Pivot such that columns are the statistics and rows are the row name,
        # and it's grouped by partition col

        pivoted = (
            current_statistics.melt(
                id_vars=[partition_key, "statistic"],
                value_vars=columns,
                var_name="column",
            )
            .pivot(
                index=[partition_key, "column"],
                columns="statistic",
                values="value",
            )
            .reset_index()
        )

        groups = []
        for _, group in pivoted.groupby(partition_key):
            groups.append(
                cls(
                    group.reset_index(drop=True),
                    string_columns,
                    float_columns,
                    int_columns,
                    partition_key,
                )
            )
        return groups

    def __str__(self) -> str:
        """
        String representation of the object's value (i.e., summary).

        Usage: `print(summary)`
        """
        return self.value.to_string()


# num_frequent_values = polars_df.groupby(partition_key).agg(
#     [
#         pl.col(c)
#         .apply(
#             lambda x: np.histogram(x, bins="auto", density=True)[
#                 0
#             ].max()
#         )
#         .alias(c)
#         for c in float_columns + int_columns
#     ]
# )

# num_frequent_values = pl.DataFrame(
#     raw_data.groupby(partition_key)
#     .agg(
#         {
#             c: lambda x: np.histogram(x, bins="auto", density=True)[
#                 0
#             ].max()
#             for c in float_columns + int_columns
#         }
#     )
#     .reset_index()
# )
