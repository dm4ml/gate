import numpy as np
import pandas as pd
import duckdb
import typing

import polars as pl


# def compute_coverage(
#     df: pl.DataFrame,
#     partition_column: str,
# ) -> pl.DataFrame:
#     """This function computes the coverage of columns in a dataframe.
#     Args:
#         df (pl.DataFrame): Dataframe to compute coverage for.
#         partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

#     Returns:
#         pl.DataFrame: Dataframe containing the coverage of each column in the dataframe, partitioned by the partition col.
#     """
#     query = f"SELECT {partition_column}, "
#     for column in columns:
#         query += f"COUNT(NULLIF({column}, NULL))::FLOAT / COUNT(*)::FLOAT AS {column}, "
#     query = query[:-2] + f" FROM {df_name} GROUP BY {partition_column}"

#     df.groupby([partition_column, variable_column]).agg(fraction_non_null=('value_col', lambda x: pl.col(x).is_not_null().mean()))

#     return con.execute(query).fetchdf()


def compute_means(
    con: duckdb.DuckDBPyConnection,
    df_name: str,
    columns: typing.List[str],
    partition_column: str,
) -> pd.DataFrame:
    """This function computes the mean of columns in a dataframe.
    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection to use.
        df_name (str): Name of the dataframe to compute mean for.
        columns (typing.List[str]): List of columns to compute mean for.
        partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

    Returns:
        pd.DataFrame: Dataframe containing the mean of each column in the dataframe, partitioned by the partition col.
    """
    query = f"SELECT {partition_column}, "
    for column in columns:
        query += f"AVG({column}) AS {column}, "
    query = query[:-2] + f" FROM {df_name} GROUP BY {partition_column}"

    return con.execute(query).fetchdf()


def compute_stdev(
    con: duckdb.DuckDBPyConnection,
    df_name: str,
    columns: typing.List[str],
    partition_column: str,
) -> pd.DataFrame:
    """This function computes the standard deviation of columns in a dataframe.
    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection to use.
        df_name (str): Name of the dataframe to compute stdev for.
        columns (typing.List[str]): List of columns to compute stdev for.
        partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

    Returns:
        pd.DataFrame: Dataframe containing the stdev of each column in the dataframe, partitioned by the partition col.
    """
    query = f"SELECT {partition_column}, "
    for column in columns:
        query += f"STDDEV_POP({column}) AS {column}, "
    query = query[:-2] + f" FROM {df_name} GROUP BY {partition_column}"

    return con.execute(query).fetchdf()


def compute_num_unique_values(
    con: duckdb.DuckDBPyConnection,
    df_name: str,
    columns: typing.List[str],
    partition_column: str,
) -> pd.DataFrame:
    """This function computes the approx number of unique values of columns in a dataframe.
    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection to use.
        df_name (str): Name of the dataframe to compute num_unique_values for.
        columns (typing.List[str]): List of columns to compute num_unique_values for.
        partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

    Returns:
        pd.DataFrame: Dataframe containing the num_unique_values of each column in the dataframe, partitioned by the partition col.
    """
    query = f"SELECT {partition_column}, "
    for column in columns:
        query += f"approx_count_distinct({column}) AS {column}, "
    query = query[:-2] + f" FROM {df_name} GROUP BY {partition_column}"

    return con.execute(query).fetchdf()


def compute_occurrence_ratio(
    con: duckdb.DuckDBPyConnection,
    df_name: str,
    columns: typing.List[str],
    partition_column: str,
) -> pd.DataFrame:
    """This function computes the ratio of the most frequently occurring value in a dataframe.
    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection to use.
        df_name (str): Name of the dataframe to compute occurrence ratio for.
        columns (typing.List[str]): List of columns to compute occurrence ratio for.
        partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

    Returns:
        pd.DataFrame: Dataframe containing the occurrence ratio of each column in the dataframe, partitioned by the partition col.
    """
    query = f"SELECT {partition_column}, "
    for column in columns:
        query += "MAX(cnt) * 1.0 / SUM(cnt) AS " + column + ", "
    query = query[:-2] + " FROM ("
    for column in columns:
        query += (
            f"SELECT {partition_column}, "
            + column
            + f", COUNT(*) AS cnt FROM {df_name} GROUP BY {partition_column}, "
            + column
            + " UNION ALL "
        )
    query = query[:-10] + f") t GROUP BY {partition_column}"

    return con.execute(query).fetchdf()


def compute_num_frequent_values(
    df: pd.DataFrame,
    columns: typing.List[str],
    partition_column: str,
) -> pd.DataFrame:
    """This function computes the count of the bin with the most values for each column.

    Args:
        df: Dataframe to compute num_frequent_values for.
        columns (typing.List[str]): List of columns to compute num_frequent_values for.
        partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

    Returns:
        pd.DataFrame: Dataframe containing the num_frequent_values of each column in the dataframe, partitioned by the partition col.
    """

    grouped = df.groupby(partition_column).agg(
        {
            c: lambda x: np.histogram(x, bins="auto", density=True)[0].max()
            for c in columns
        }
    )
    return grouped.reset_index()


def type_to_statistics(t: str) -> typing.List[str]:
    """Returns the statistics that can be computed for a given type.

    Args:
        t (str): Type (one of "int", "float", "string").

    Returns:
        typing.List[str]: List of statistics that can be computed for the type.

    Raises:
        ValueError: If the type is unknown.
    """

    if t == "int":
        return [
            "coverage",
            "mean",
            "stdev",
            "num_unique_values",
            "occurrence_ratio",
            "p95",
        ]

    if t == "float":
        return [
            "coverage",
            "mean",
            "stdev",
            "p95",
        ]

    if t == "string":
        return ["coverage", "num_unique_values", "occurrence_ratio"]

    raise ValueError(f"Unknown type {t}")
