GATE is a Python module that detects drift in partitions of data. GATE computes partition _summaries_, which are then fed into an anomaly detection algorithm to detect whether a new partition is anomalous. This minimizes false positive alerts when detecting drift in machine learning (ML) pipelines, where there may be many features and prediction columns.

!!! tip "Support for Embeddings"

    We now support drift detection on embeddings, in addition to structured data. GATE considers _both_ the structured data and the embeddings when computing partition summaries and detecting drift. Check out the [embeddings page](./embedding) for a walkthrough of how to use GATE with embeddings.

## Installation

GATE is available on PyPI and can be installed with pip:

```bash
pip install gate-drift
```

Note that GATE requires Python 3.8 or higher.

## Usage

GATE is designed to be used with [Pandas](https://pandas.pydata.org/) dataframes. Check out the [example](./example) for a walkthrough of how to use GATE.

## Research Contributions

GATE was developed and is maintained by researchers at the UC Berkeley [EPIC Lab](https://epic.berkeley.edu/).

An initial version of GATE was developed as part of a collaboration with Meta, and the research paper, "Moving Fast With Broken Data" by Shankar et al., is available on [arXiv](https://arxiv.org/abs/2303.06094). This module slightly differs from the original implementation, but the core ideas around partition summaries and anomaly detection are the same.
