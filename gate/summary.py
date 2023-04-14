import duckdb

import pandas as pd
import typing

from gate.statistics import (
    compute_coverage,
    compute_means,
    compute_stdev,
    compute_num_unique_values,
    compute_num_frequent_values,
)


class Summary(object):
    def __init__(
        self,
        value: pd.DataFrame,
        string_columns: typing.List[str],
        float_columns: typing.List[str],
        int_columns: typing.List[str],
        partition_column: str,
        window: int,
    ):
        self._value = value
        self._string_columns = string_columns
        self._float_columns = float_columns
        self._int_columns = int_columns
        self._partition_column = partition_column
        self._window = window

    @property
    def value(self) -> pd.DataFrame:
        return self._value

    @property
    def columns(self) -> typing.List[str]:
        return self._string_columns + self._float_columns + self._int_columns

    @property
    def partition_column(self) -> str:
        return self._partition_column

    @property
    def window(self) -> int:
        return self._window

    @classmethod
    def fromRaw(
        cls,
        raw_data: pd.DataFrame,
        columns: typing.List[str] = [],
        partition_column: str = "",
        window: int = 0,
        previous_summaries: typing.List["Summary"] = [],
    ) -> typing.List["Summary"]:
        if len(previous_summaries) > 0:
            partition_column = previous_summaries[0].partition_column
            columns = previous_summaries[0].columns
            string_columns = previous_summaries[0]._string_columns
            float_columns = previous_summaries[0]._float_columns
            int_columns = previous_summaries[0]._int_columns
            window = previous_summaries[0].window
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
            float_columns = [
                c for c, t in column_types.items() if t == "float"
            ]
            int_columns = [c for c, t in column_types.items() if t == "int"]
            assert len(string_columns) + len(float_columns) + len(
                int_columns
            ) == len(columns)

        if partition_column not in raw_data.columns:
            raise ValueError(
                f"Partition column {partition_column} is not in dataframe columns."
            )
        if not set(columns).issubset(set(raw_data.columns)):
            raise ValueError(
                "Columns to compute summaries on are not all in the dataframe."
            )

        # Compute the summary statistics
        con = duckdb.connect()
        con.register("raw_data", raw_data)

        statistics = {
            "coverage": compute_coverage(
                con, "raw_data", columns, partition_column
            ),
            "mean": compute_means(
                con, "raw_data", float_columns + int_columns, partition_column
            ),
            "stdev": compute_stdev(
                con, "raw_data", float_columns + int_columns, partition_column
            ),
            "num_unique_values": compute_num_unique_values(
                con, "raw_data", string_columns + int_columns, partition_column
            ),
            # "occurrence_ratio": compute_occurrence_ratio(
            #     con, "raw_data", columns, partition_column
            # ),
            "occurrence_ratio": raw_data.groupby(partition_column)
            .apply(
                lambda x: x.apply(
                    lambda y: y.value_counts(normalize=True).iloc[0]
                )[columns]
            )
            .reset_index(),
            "num_frequent_values": compute_num_frequent_values(
                con, "raw_data", float_columns + int_columns, partition_column
            ),
        }

        # Merge the statistics into a single dataframe
        for name, df in statistics.items():
            df["statistic"] = name
            statistics[name] = df

        current_statistics = pd.concat(statistics.values(), ignore_index=True)

        # Pivot such that columns are the statistics and rows are the row name, and it's grouped by partition col

        pivoted = (
            current_statistics.melt(
                id_vars=[partition_column, "statistic"],
                value_vars=columns,
                var_name="column",
            )
            .pivot_table(
                index=[partition_column, "column"],
                columns="statistic",
                values="value",
            )
            .reset_index()
        )

        groups = []
        for _, group in pivoted.groupby(partition_column):
            groups.append(
                cls(
                    group.reset_index(drop=True).copy(),
                    string_columns,
                    float_columns,
                    int_columns,
                    partition_column,
                    window,
                )
            )
        return groups

    def __str__(self):
        return self.value.to_string()
