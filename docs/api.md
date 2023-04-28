::: gate.summarize
    handler: python
    options:
      members:
        - summarize
        - compute_embeddings
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false

::: gate.summary.Summary
    handler: python
    options:
      members:
        - summary
        - embeddings_summary
        - partition_key
        - partition
        - columns
        - non_embedding_columns
        - embedding_examples
        - embedding_centroids
        - statistics
        - value
        - __str__
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: true
      show_source: false

::: gate.drift
    handler: python
    options:
      members:
        - detect_drift
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false

::: gate.drift.DriftResult
    handler: python
    options:
      members:
        - summary
        - neighbor_summaries
        - drifted_examples
        - score
        - score_percentile
        - is_drifted
        - all_scores
        - clustering
        - drill_down
        - drifted_columns
        - __str__
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: true
      show_source: false

::: gate.statistics
    handler: python
    options:
      members:
        - type_to_statistics
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false