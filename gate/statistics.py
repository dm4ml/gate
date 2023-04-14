import pandas as pd
import duckdb
import typing


def compute_coverage(
    con: duckdb.DuckDBPyConnection,
    df_name: str,
    columns: typing.List[str],
    partition_column: str,
) -> pd.DataFrame:
    """This function computes the coverage of columns in a dataframe.
    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection to use.
        df_name (str): Name of the dataframe to compute coverage for.
        columns (typing.List[str]): List of columns to compute coverage for.
        partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

    Returns:
        pd.DataFrame: Dataframe containing the coverage of each column in the dataframe, partitioned by the partition col.
    """
    query = f"SELECT {partition_column}, "
    for column in columns:
        query += f"COUNT(NULLIF({column}, NULL))::FLOAT / COUNT(*)::FLOAT AS {column}, "
    query = query[:-2] + f" FROM {df_name} GROUP BY {partition_column}"

    return con.execute(query).fetchdf()


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
    con: duckdb.DuckDBPyConnection,
    df_name: str,
    columns: typing.List[str],
    partition_column: str,
) -> pd.DataFrame:
    """This function computes the count of the bin with the most values for each column.

    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection to use.
        df_name (str): Name of the dataframe to compute num_frequent_values for.
        columns (typing.List[str]): List of columns to compute num_frequent_values for.
        partition_column (str): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition.

    Returns:
        pd.DataFrame: Dataframe containing the num_frequent_values of each column in the dataframe, partitioned by the partition col.
    """
    queries = []
    for col in columns:
        queries.append(f"histogram({col}) AS {col}")

    query = f"SELECT {partition_column}, {''.join([f'{q},' for q in queries])[:-1]} FROM {df_name} GROUP BY {partition_column}"

    result = con.execute(query).fetchdf()
    new_result = pd.DataFrame(columns=result.columns)

    for col in result.columns:
        if col == partition_column:
            new_result[col] = result[col]
        else:
            new_result[col] = result[col].apply(lambda x: max(x["value"]))

    return new_result
