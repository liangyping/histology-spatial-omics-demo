# Histology-to-Spatial Omics Prediction Demo

This repository contains a simplified demo workflow for predicting spatial gene expression patterns from H&E histopathology images.

The original research project is still under active development. For this reason, this repository only includes a lightweight code structure, example scripts, and evaluation utilities. It does not include private clinical data, raw spatial transcriptomics data, trained model weights, or unpublished core implementation details.

## Background

Spatial transcriptomics can provide gene expression profiles together with tissue spatial information, but it is still difficult to apply at large scale in routine clinical settings. In contrast, H&E-stained histopathology slides are widely available in pathology workflows.

This demo explores a computational workflow for learning associations between histological morphology and spatial gene expression patterns. The goal is to show the general structure of a histology-to-spatial-omics prediction pipeline, rather than to release the full research system.

## What is included

* Basic preprocessing utilities for histology image patches
* Example formatting steps for spatial omics data
* Simple graph-neighborhood construction utilities
* A simplified model skeleton
* Evaluation metrics such as PCC, Spearman correlation, and RMSE
* Demo scripts for preprocessing and inference-style evaluation

## Repository Structure

```text
histology-spatial-omics-demo/
├── README.md
├── requirements.txt
├── configs/
│   └── example_config.yaml
├── src/
│   ├── data_utils.py
│   ├── graph_utils.py
│   ├── metrics.py
│   └── model_skeleton.py
├── demo/
│   ├── demo_preprocessing.py
│   └── demo_inference.py
└── figures/
    └── workflow.png
```

## Quick Start

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the demo preprocessing script:

```bash
python demo/demo_preprocessing.py --config configs/example_config.yaml
```

Run the demo evaluation script:

```bash
python demo/demo_inference.py --config configs/example_config.yaml
```

The demo scripts are designed to illustrate the workflow structure. They may need to be adapted to specific datasets and local file paths.

## Evaluation

The demo provides common evaluation metrics for spatial gene expression prediction, including:

* Pearson correlation coefficient (PCC)
* Spearman correlation
* RMSE
* Gene-level evaluation utilities such as HEG, HVG, and marker gene subsets

The metric functions are implemented in `src/metrics.py`.

## Data Note

No patient data or raw research data are included in this repository.

For public spatial transcriptomics datasets, users should follow the license and usage requirements of the original data source. Local paths and dataset-specific preprocessing steps should be configured in `configs/example_config.yaml`.

## Limitations

This repository is intended as a simplified academic demo. It is not a full reproduction of the ongoing research project.

The following materials are not included:

* Private clinical data
* Original whole-slide images
* Raw spatial transcriptomics files
* Trained model checkpoints
* Unpublished model modules
* Patent-related implementation details

## Purpose

This repository is mainly used to demonstrate project organization, preprocessing logic, evaluation design, and reproducible coding practices for computational pathology and spatial omics research.

## License

This demo is provided for academic and educational purposes only.
