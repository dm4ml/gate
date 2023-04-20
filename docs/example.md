There are two functions exposed by the GATE module: [`summarize`](/gate/api/#gate.summarize.summarize) and [`detect_drift`](/gate/api/#gate.drift.detect_drift). [`summarize`](/gate/api/#gate.summarize.summarize) computes partition summaries for a dataframe, and [`detect_drift`](/gate/api/#gate.drift.detect_drift) detects whether a new partition is drifted. 

In this example, we'll demonstrate how to use GATE to detect drift in small synthetic dataset. 

## Dataset Creation

Our synthetic dataset will be created in Pandas. The partition key will be `date`. There will be 10 partitions, and each partition will have 10,000 rows. There will be 3 columns. The last partition will have a different column distribution than the first 9 partitions.

```python
import numpy as np
import pandas as pd

# create example date range
date_range = pd.date_range(start="2022-01-01", periods=10, freq="D")

# create example data for each column
int_col = np.random.randint(low=0, high=10, size=10000)
float_col = np.random.normal(loc=0, scale=1, size=10000)
string_col = np.random.choice(["A", "B", "C"], size=10000)

# combine data into a DataFrame
df_elems = []
for date in date_range:
    date_data = {"date": date}
    if date != date_range[-1]:
        date_data = pd.DataFrame(
            {
                "date": [date] * len(int_col),
                "int_col": int_col,
                "float_col": float_col,
                "string_col": string_col,
            }
        )
    else:
        # Change the distribution of the int column
        date_data = pd.DataFrame(
            {
                "date": [date] * len(int_col),
                "int_col": np.random.randint(low=10, high=20, size=10000),
                "float_col": float_col,
                "string_col": string_col
            }
        )
    df_elems.append(date_data)

df = pd.concat(df_elems).reset_index(drop=True)
```

## [`summarize`](/gate/api/#gate.summarize.summarize)

The [`summarize`](/gate/api/#gate.summarize.summarize) function computes partition summaries for a dataframe. In addition to a Pandas dataframe of raw data, it accepts the partition key and a list of columns in the dataframe to compute statistics for. Or, one can specify a list of previous partition summaries instead of the partition key and column list, and GATE will infer the partition key and columns from the previous partition summaries.

The [`summarize`](/gate/api/#gate.summarize.summarize) function returns a list of [`Summary`](/gate/api/#gate.summary.Summary) objects. Each [`Summary`](/gate/api/#gate.summary.Summary) object contains the partition summary and other metadata, and has a `__str__` method that prints the summary in a human-readable format.


```python
from gate import summarize

summaries = summarize(
    df, partition_key="date", columns=["int_col", "float_col", "string_col"]
)
# len(summaries) == 10 because there are 10 distinct partitions

print(summaries[-1])

"""
statistic       date      column  coverage       mean  num_unique_values  occurrence_ratio        p95     stdev
0         2022-01-10   float_col       1.0  -0.005204                NaN               NaN   1.622783  0.995632
1         2022-01-10     int_col       1.0  14.504000               10.0            0.1051  19.000000  2.891716
2         2022-01-10  string_col       1.0        NaN                3.0            0.3388        NaN       NaN 
"""
```

!!! note

    You can access the summary data as a Pandas dataframe with the `value` attribute of the [`Summary`](/gate/api/#gate.summary.Summary) object (i.e., `summaries[-1].value`).

## [`detect_drift`](/gate/api/#gate.drift.detect_drift)

The [`detect_drift`](/gate/api/#gate.drift.detect_drift) function detects whether a new partition is drifted. It accepts a new partition summary and list of previous partition summaries and returns a [`DriftResult`](/gate/api/#gate.drift.DriftResult) object. The [`DriftResult`](/gate/api/#gate.drift.DriftResult) object has a `__str__` method that prints the drift result in a human-readable format.

```python
from gate import detect_drift

drift_result = detect_drift(summaries[-1], summaries[:-1])
print(drift_result)

"""
Drift score: 6.3246 (100.00% percentile)
Top drifted columns:
           statistic   z-score
column                        
int_col          p95  2.846050
float_col        p95  0.000002
string_col  coverage  0.000000
"""
```

The z-score represents the number of standard deviations away from the mean that the new partition is. In this case, the int col correctly has a high z-score. We recommend focusing on z-scores > 2.5 or < -2.5 when looking for drift.

If you want to cluster correlated columns, you can pass `cluster = True` into [`detect_drift`](/gate/api/#gate.drift.detect_drift). The [`DriftResult`](/gate/api/#gate.drift.DriftResult) object has a `clustering` attribute that contains the clusters.

!!! note

    The list of previous partition summaries must have at least one element. Best results are achieved when there are at least 5 previous partition summaries.

## Real Dataset Example

For an end-to-end example on a real weather dataset, see the [example notebook](https://www.github.com/dm4ml/gate/blob/main/examples/weather.ipynb) in the Github repository.