import typing

import pandas as pd
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

from gate.summary import Summary


def compute_embeddings(column: pd.Series, column_type: str) -> pd.Series:
    """Computes embeddings for a Series with the
    huggingface/transformers library. We use the
    clip-ViT-B-32 model.

    Args:
        column (pd.Series): Series to compute embeddings for.
            Must be of string type. Can contain either
            paths to files or text.
        column_type (str): Type of the column. Must be "text" or "image".

    Returns:
        pd.Series: Series of embeddings to add to your DataFrame.
    """
    assert column_type in [
        "text",
        "image",
    ], "column_type must be text or image"

    model = SentenceTransformer("clip-ViT-B-32")

    def compute_embedding_helper(text: str) -> typing.List[float]:
        if column_type == "image":
            try:
                img = Image.open(text)
                return model.encode(img)
            except FileNotFoundError:
                img = Image.open(requests.get(text, stream=True).raw)
                return model.encode(img)
            except Exception as e:
                raise e

        return model.encode(text)

    return column.apply(compute_embedding_helper)


def summarize(
    df: pd.DataFrame,
    columns: typing.List[str] = [],
    embedding_column_map: typing.Dict[str, str] = {},
    partition_key: str = "",
    previous_summaries: typing.List[Summary] = [],
) -> typing.List[Summary]:
    """This function computes partition-wide summary statistics for the given
    columns. df can have multiple partitions.

    Args:
        df (pd.DataFrame):
            Dataframe to summarize.
        columns (typing.List[str], optional):
            List of columns to generate summary statistics for. Must be a
            subset of df.columns. If empty, previous_summaries must not be
            empty.
        embedding_column_map (typing.Dict[str, str], optional):
            Dictionary of embedding key to embedding value column. Keys and
            values must be in df.columns. If empty, previous_summaries must not
            be empty.
        partition_key (str, optional):
            Name of column to partition the dataframe by. Must be in df.
            columns. Can be empty if no partitioning is desired, or if the
            dataframe represents a single partition. If empty,
            previous_summaries must not be empty.
        previous_summaries (typing.List[Summary], optional):
            List of Summary objects representing previous partition summaries.

    Returns:
        typing.List[Summary]:
            List of Summary objects, one per distinct partition found in df.

    Raises:
        ValueError:
            If `partition_key` is "group".
        ValueError:
            If `columns is empty` and `previous_summaries` is empty.
        ValueError:
            If `partition_key `is empty and `previous_summaries` is empty.
        ValueError:
            If `partition_key` is not in `df.columns`.
        ValueError:
            If any column in `columns` is not in `df.columns`.
    """
    if partition_key == "group":
        raise ValueError("Please rename the partition_key; it cannot be `group`.")

    if len(previous_summaries) == 0:
        if len(columns) == 0 and len(embedding_column_map) == 0:
            raise ValueError(
                "You must pass in some columns if you do not have any previous"
                " summaries."
            )
        if not partition_key:
            raise ValueError(
                "You must pass in a partition column if you do not have any"
                " previous summaries."
            )

    summary = Summary.fromRaw(
        df,
        columns=columns,
        embedding_column_map=embedding_column_map,
        partition_key=partition_key,
        previous_summaries=previous_summaries,
    )

    return summary
