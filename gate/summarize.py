import pandas as pd
import typing

from gate.summary import Summary


def summarize(
    df: pd.DataFrame,
    columns: typing.List[str] = [],
    partition_column: str = "",
    window: int = 0,
    previous_summaries: typing.List[Summary] = [],
) -> typing.List[Summary]:
    """This function computes partition-wise summary statistics for the given columns.

    Args:
        df (pd.DataFrame): Dataframe to summarize.
        columns (typing.List[str], optional): List of columns to generate summary statistics for. Must be a subset of df.columns. If empty, previous_summaries must not be empty.
        partition_column (str, optional): Column to partition the dataframe by. Must be in df.columns. Can be empty if no partitioning is desired, or if the dataframe represents a single partition. If empty, previous_summaries must not be empty.
        window (int, optional): Window size to use for computing rolling statistics. Defaults to 0, which means no rolling statistics are computed.
        previous_summaries (typing.List[Summary], optional): list of Summary objects representing previous partition summaries.

    Returns:
        typing.List[Summary]: List of Summary objects, one per distinct partition found in df.
    """
    if partition_column == "group":
        raise ValueError(
            "Please rename the partition_column; it cannot be `group`."
        )

    if len(previous_summaries) == 0:
        if len(columns) == 0:
            raise ValueError(
                f"You must pass in some columns if you do not have any previous summaries."
            )
        if not partition_column:
            raise ValueError(
                f"You must pass in a partition column if you do not have any previous summaries."
            )

    summary = Summary.fromRaw(
        df,
        columns=columns,
        partition_column=partition_column,
        window=window,
        previous_summaries=previous_summaries,
    )

    return summary
