# DRP Ensemble Prediction Pipeline

This repository hosts a powerful and flexible Nextflow pipeline for Drug Response Prediction (DRP). It integrates a curated set of state-of-the-art models, allowing users to either run model groups individually or execute a complete, sequential workflow that culminates in a robust ensemble prediction.

This version represents a significant update, streamlining the model set and optimizing the execution flow with Nextflow profiles.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Method 1: Full Ensemble Workflow (Recommended)](#method-1-full-ensemble-workflow-recommended)
  - [Method 2: Modular Execution (Advanced)](#method-2-modular-execution-advanced)
- [Workflow Parameters](#workflow-parameters)
  - [Workflow Control](#workflow-control)
  - [GPU Allocation](#gpu-allocation)
  - [Core Input Files](#core-input-files)
  - [Ensemble Analysis Parameters](#ensemble-analysis-parameters)
- [Project Structure](#project-structure)
- [Example Commands](#example-commands)

## Overview

Predicting the response of cancer cell lines to various drugs is a critical task in computational biology. This pipeline orchestrates the execution of multiple DRP models, handling their unique dependencies and computational requirements. It offers two main modes of operation: a one-click script (`run_ensemble.sh`) for a full, sequential analysis, and a modular approach for running specific model groups or the final analysis on demand. The final predictions from each model are aggregated by an `Ensemble Analysis` module to generate consensus results.

## Features

-   **One-Click Ensemble Script**: Includes `run_ensemble.sh` to automate the entire process: run all model groups sequentially and then generate the final ensemble prediction in an optimized order.
-   **Granular GPU Allocation**: Assign distinct GPU IDs to different model groups to optimize resource usage on multi-GPU systems.
-   **Modular Design**: Each DRP model group is encapsulated in its own Nextflow sub-workflow, enabling flexible execution via profiles (`part1`, `dipk_graphdrp`, `voter`).
-   **Integrated Ensemble Analysis**: A powerful module to create an ensemble prediction from the results of individual models using machine learning or weighted averaging.
-   **Reproducibility & Scalability**: Leverages Nextflow's capabilities to create reproducible, scalable, and fault-tolerant computational workflows.
-   **Test Data Included**: Comes with a sample dataset for immediate, out-of-the-box testing.

## Prerequisites

-   **Nextflow**: The pipeline is built on Nextflow (v21.10.x or later recommended).
    ```bash
    curl -s https://get.nextflow.io | bash
    # Move the 'nextflow' executable to a directory in your $PATH
    ```
-   **NVIDIA GPU**: The deep learning models are computationally intensive and require an NVIDIA GPU with appropriate CUDA drivers installed.
-   **Bash Environment**: A standard Bash shell is required to use the `run_ensemble.sh` script.
-   **Conda**: Conda is used for environment management. Nextflow will automatically create the necessary environments from the `.yml` files.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Make the script executable:**
    ```bash
    chmod +x run_ensemble.sh
    ```

### Method 1: Full Ensemble Workflow (Recommended)

This is the simplest way to get a complete result. The `run_ensemble.sh` script runs all model workflows in a specific, optimized order, cleaning up intermediate files between steps. 

**Note**: This script now uses a fixed `./results` directory for all outputs, unlike previous versions which created timestamped folders.

-   **Run with the built-in test data:**
    ```bash
    ./run_ensemble.sh
    ```

-   **Run with your own data and custom GPU allocation:**
    You can pass any Nextflow parameter directly to the script. These will be automatically forwarded to all underlying pipeline runs.
    ```bash
    ./run_ensemble.sh --gene_exp_file /path/to/exp.csv \
                      --mutation_file /path/to/mut.csv \
                      --gpu_part1 0 --gpu_dipk_graphdrp 1
    ```
    All results will be stored in the `./results/` directory.

### Method 2: Modular Execution (Advanced)

If you only want to run a specific model group or the ensemble analysis, you can call `main.nf` directly using profiles defined in `nextflow.config`. Results will be saved to the directory specified by `--output_dir` (default: `./results`).

-   **Run the `part1` model group:**
    ```bash
    nextflow run main.nf -profile part1 --gpu_part1 0
    ```

-   **Run the `DIPK-GraphDRP` workflow:**
    ```bash
    nextflow run main.nf -profile dipk_graphdrp --gpu_dipk_graphdrp 1
    ```

-   **Run the Ensemble Analysis on existing results:**
    This should be run after one or more model workflows have completed and their `*_predictions.csv` files are present in the output directory.
    ```bash
    nextflow run main.nf -profile voter --output_dir ./results
    ```

## Workflow Parameters

Parameters can be specified on the command line (e.g., `--gpu_part1 1`).

### Workflow Control

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `entry` | Specifies the workflow to run. Used internally by profiles. Options: `part1`, `dipk_graphdrp`, `voter`. | `''` |
| `output_dir` | The root directory where all prediction results will be saved. | `${projectDir}/results` |

### GPU Allocation

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `gpu_part1` | GPU ID for the `part1` model group (BANDRP, DeepTTC, etc.). | `0` |
| `gpu_dipk_graphdrp` | GPU ID for the `DIPK` & `GraphDRP` model group. | `0` |

### Core Input Files

| Parameter | Description | Default (points to test data) |
| :--- | :--- | :--- |
| `drug_smiles` | CSV file with drug SMILES strings. | `${projectDir}/test/drug_sample.csv` |
| `gene_exp_file` | Gene expression data file. | `${projectDir}/test/gene_sample.csv` |
| `mutation_file` | Gene mutation data file. | `${projectDir}/test/mu_sample.csv` |
| `cnv_file` | Copy Number Variation (CNV) data file. | `${projectDir}/test/cnv_sample.csv` |
| `gsva_file` | GSVA analysis results file. | `${projectDir}/test/gsva_sample.csv` |
| `cell_file_graphdrp`| Cell line features file for GraphDRP. | `${projectDir}/test/mu_sample.csv` |

### Ensemble Analysis Parameters

These parameters are used by the final `voter` workflow which runs the `run_ensemble.py` script.

*Note: The previous method of enabling/disabling models with flags (e.g., `--BANDRP 1`) has been deprecated. The new ensemble script automatically discovers prediction files in the output directory.*

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `model_pkl` | Path to the trained RandomForest model (`.pkl`) used by the ensemble script in `mode 0`. | `${projectDir}/best_model_RandomForest_rmse.pkl` |
| `weight_file` | Path to the `weight.csv` file defining model weights for the ensemble script in `mode 1`. | `${projectDir}/weight.csv` |

## Project Structure