::: gate.summarize
    handler: python
    options:
      members:
        - summarize
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false

::: gate.summary.Summary
    handler: python
    options:
      members:
        - value
        - partition_key
        - partition
        - columns
        - statistics
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
        - is_drifted
        - score
        - score_percentile
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