import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from gate.summary import Summary
import typing

from sklearn.neighbors import NearestNeighbors


class DriftResult(object):
    def __init__(self, all_scores: pd.Series, nn_features: pd.DataFrame):
        self._all_scores = all_scores
        self._nn_features = nn_features

    @property
    def score(self):
        return self._all_scores.iloc[-1]

    @property
    def score_percentile(self):
        # Check what percentile the last elem of preds is
        return percentileofscore(self.all_scores, self.score) * 1.0 / 100

    @property
    def all_scores(self):
        return self._all_scores

    def drill_down(self) -> pd.DataFrame:
        # Return a dataframe with features with highest magnitude anomaly
        # scores

        last_day = self._nn_features.iloc[-1]
        sorted_cols = last_day.abs().sort_values(ascending=False).index
        sorted_df = last_day[sorted_cols].to_frame()
        sorted_df.rename(
            columns={sorted_df.columns[0]: "z-score"}, inplace=True
        )

        return sorted_df

    def drifted_columns(self, limit: int = 10) -> pd.DataFrame:
        # Return a dataframe of the top limit columns that have drifted
        # Drop duplicate column names
        dd_results = self.drill_down()

        dd_results.reset_index(inplace=True)
        dd_results.drop_duplicates(
            subset=["column"], keep="first", inplace=True
        )
        dd_results.set_index("column", inplace=True)

        return dd_results.head(limit)


def detect_drift(
    current_summary: Summary,
    previous_summaries: typing.List[Summary],
    validity: typing.List[int] = [],
    cluster: bool = True,
    k: int = 5,
) -> DriftResult:
    """Computes whether the current partition summary has drifted from previous summaries.

    Args:
        current_summary (Summary): Partition summary for current partition.
        previous_summaries (typing.List[Summary]): Previous partitions' summaries.
        validity (typing.List[int], optional): indicator list identifying which partition summaries are valid. 1 if valid, 0 if invalid. If empty, we assume all partition summaries are valid. Must be empty or equal to length of previous_summaries.
        cluster (bool, optional): Whether or not to cluster columns in summaries. Increases runtime but also increases precision in drift detection. Only engaged if summaries have more than 10 columns. Defaults to True.
        k (int, optional): Number of nearest neighbor partitions to inspect.

    Returns (DriftResult): DriftResult object with score and score percentile.
    """
    partition_column = current_summary.partition_column
    columns = current_summary.columns
    statistics = current_summary.statistics

    # Create validity vector
    if not validity:
        validity = [1] * len(previous_summaries)
    if len(validity) != len(previous_summaries):
        raise ValueError(
            f"Validity vector has length {len(validity)} but should have length {len(previous_summaries)} to match previous_summaries."
        )
    validity.append(1)

    # Normalize current and previous partition summaries
    all_summaries = pd.concat(
        [p.value for p in previous_summaries] + [current_summary.value]
    ).reset_index(drop=True)
    normalized = all_summaries.copy()
    normalized.set_index([partition_column, "column"], inplace=True)

    groups = normalized.groupby("column")
    mean, std = groups.transform("mean"), groups.transform("std")
    std += 1e-7
    normalized = (normalized[mean.columns] - mean) / std
    normalized.reset_index(inplace=True)
    normalized = normalized.fillna(0.0)

    # Run nearest neighbor algorithm to get distances
    nn_features = normalized.pivot_table(
        index=partition_column, columns="column", values=statistics
    )
    knn = NearestNeighbors(
        n_neighbors=len(nn_features),
        p=2,
        n_jobs=-1,
    )
    knn.fit(nn_features)
    neighbor_graph = knn.kneighbors_graph(
        nn_features,
        n_neighbors=len(nn_features),
        mode="distance",
    ).todense()

    for v in validity:
        if v == 0:
            v = np.nan

    tril = np.tril(neighbor_graph)
    tril = tril * np.vstack([validity for _ in range(len(validity))])
    # tril[tril == 0] = np.nan

    # Take mean of minimum k elements, row-wise
    mink = np.sort(tril, axis=1)
    mink[:, k:] = np.nan

    scores = pd.Series(
        data=np.nanmean(mink, axis=1),
        index=nn_features.index,
    )

    dr = DriftResult(scores, nn_features)

    return dr
