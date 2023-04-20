# GATE: Data Drift Detection for Machine Learning Pipelines

[![GATE](https://github.com/dm4ml/gate/workflows/gate/badge.svg)](https://github.com/dm4ml/gate/actions?query=workflow:"gate")
[![lint (via ruff)](https://github.com/dm4ml/gate/workflows/lint/badge.svg)](https://github.com/dm4ml/gate/actions?query=workflow:"lint")
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

GATE is a Python module that detects drift in partitions of data. GATE computes partition summaries, which are then fed into an anomaly detection algorithm to detect whether a new partition is anomalous. This minimizes false positive alerts when detecting drift in machine learning (ML) pipelines, where there may be many features and prediction columns.

## Installation

GATE is available on PyPI and can be installed with pip:

```bash
pip install gate-drift
```

## Usage

GATE is designed to be used with [Pandas](https://pandas.pydata.org/) dataframes. Check out the [documentation](https://dm4ml.github.io/gate/) for a walkthrough of how to use GATE.

## Research Contributions

GATE was developed and is maintained by researchers at the UC Berkeley [EPIC Lab](https://epic.berkeley.edu/).

An initial version of GATE was developed as part of a collaboration with Meta, and the research paper, "Moving Fast With Broken Data" by Shankar et al., is available on [arXiv](https://arxiv.org/abs/2303.06094). This module slightly differs from the original implementation, but the core ideas around partition summaries and anomaly detection are the same.
