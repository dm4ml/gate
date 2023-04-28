# Drift Detection on Embeddings

GATE supports drift detection and debugging of embeddings, in addition to structured data. At a high level, embeddings are represented in their own column, and you can call [`summarize`](/gate/api/#gate.summarize.summarize) and [`detect_drift`](/gate/api/#gate.drift.detect_drift) on dataframes with embedding columns.

## Embedding key and value columns

In your original dataframe, you should have a column that contains the embedding key, and a column that contains the embedding value. The key column should be a string (e.g., text, filename), and the value column should be a list of floats. For example:

```python
df = pd.DataFrame(
    {
        "date": ["2020-01-01", "2020-01-01", "2020-01-01"], # This is the partition key
        "text": ["hello world!", "goodbye", "a third greeting"],
        "embedding": [
            [0.1, 0.2, 0.3], # Imagine this is the embedding for "hello world!"
            [0.4, 0.5, 0.6], # Imagine this is the embedding for "goodbye"
            [0.7, 0.8, 0.9], # Imagine this is the embedding for "a third greeting"
        ],
    }
)
```

Then, when calling [`summarize`](/gate/api/#gate.summarize.summarize) on your dataframe, you can specify the embedding key-value pairs as follows:

```python
from gate import summarize

summarize(
    df,
    partition_key="date",
    embedding_column_map={"text": "embedding"},
)
```

Both keys and values in `embedding_column_map` should be strings, representing column names in your dataframe.

## Summarizing embeddings

When you call [`summarize`](/gate/api/#gate.summarize.summarize) on a dataframe with embedding columns, GATE will automatically compute summary statistics for each dimension in the embedding values. You can access these summaries by calling [`embeddings_summary`](/gate/api/#gate.summary.Summary.embeddings_summary) on the returned [`Summary`](/gate/api/#gate.summary.Summary) object.

GATE will also cluster the embeddings, compute centroids for each cluster, and store examples for each cluster. Embeddings are clustered for each embedding column separately. You can access the examples by calling [`embedding_examples`](/gate/api/#gate.summary.Summary.embedding_examples) on the returned [`Summary`](/gate/api/#gate.summary.Summary) object. You can access the centroids by calling [`embedding_centroids`](/gate/api/#gate.summary.Summary.embedding_centroids) on the returned [`Summary`](/gate/api/#gate.summary.Summary) object.


```python
from gate import summarize

summaries = summarize(
    df,
    partition_key="date",
    columns=[], # No structured columns
    embedding_column_map={"text": "embedding"},
) # (1)!

# Get the summary statistics for the embedding values
summaries[0].embeddings_summary

# Get the examples for each cluster
summaries[0].embedding_examples("text") # Must passing embedding key

# Get the centroids for each cluster
summaries[0].embedding_centroids("text") # Must passing embedding key
```

1. Note that [`summarize`](/gate/api/#gate.summarize.summarize) returns a list of [`Summary`](/gate/api/#gate.summary.Summary) objects, one for each partition key. In this example, we only have one partition key, so we access the first element of the list.

In practice, you probably won't need to call [`embedding_examples`](/gate/api/#gate.summary.Summary.embedding_examples) or [`embedding_centroids`](/gate/api/#gate.summary.Summary.embedding_centroids) directly. These methods are used in `detect_drift`, as described below.

## Detecting drift on embeddings

You can call [`detect_drift`](/gate/api/#gate.drift.detect_drift) on summaries of dataframes with embedding columns. Drift detection takes both structured column data and embeddings into consideration, if you have both.

[`detect_drift`](/gate/api/#gate.drift.detect_drift) will return a [`DriftResult`](/gate/api/#gate.drift.DriftResult) object, which contains the following information relevant to embeddings:

- [`drifted_columns`](/gate/api/#gate.drift.DriftResult.drifted_columns): Returns a dataframe of column names that have drifted, their most anomalous statistic (e.g., coverage), and the z-score. This includes both structured columns and embedding columns.
- [`drifted_examples`](/gate/api/#gate.drift.DriftResult.drifted_examples): Returns examples that have drifted most from their historical clusters. This is specific to embeddings. The object returned is a dictionary with `drifted_examples` and `corresponding_examples` keys. The value of each key is a dataframe with columns `partition_key`, `embedding_key_column`, and `embedding_value_column`.

An example of calling [`detect_drift`](/gate/api/#gate.drift.detect_drift) on a dataframe with embedding columns is shown below:

```python
from gate import detect_drift

drift_result = detect_drift(
    summary,
    previous_summaries
)

# Get the drifted columns
drift_result.drifted_columns()

# Get the drifted examples
drifted_example_result = drift_result.drifted_examples("text") # Must passing embedding key
drifted_example_result["drifted_examples"]
```

## Real Dataset Example

For an example of using GATE with embeddings, see this [example notebook](https://www.github.com/dm4ml/gate/blob/main/examples/civilcomments.ipynb) in the Github repository.