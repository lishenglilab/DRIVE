# DRP Ensemble Prediction Pipeline

This repository hosts a powerful and flexible Nextflow pipeline for Drug Response Prediction (DRP). It integrates over 15 state-of-the-art models, allowing users to either run them individually or execute a complete, sequential workflow that culminates in a robust ensemble prediction.

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
  - [Ensemble Voter Parameters](#ensemble-voter-parameters)
- [Project Structure](#project-structure)
- [Example Commands](#example-commands)

## Overview

Predicting the response of cancer cell lines to various drugs is a critical task in computational biology. This pipeline orchestrates the execution of multiple DRP models, handling their unique dependencies and computational requirements. It offers two main modes of operation: a one-click script (`run_ensemble.sh`) for a full, sequential analysis, and a modular approach for running specific models or groups on demand. The final predictions from each model can be aggregated by a `Voter` module to generate a consensus result.

## Features

-   **One-Click Ensemble Script**: Includes `run_ensemble.sh` to automate the entire process: run all models sequentially and then generate the final ensemble prediction.
-   **Granular GPU Allocation**: Assign distinct GPU IDs to different model groups to optimize resource usage on multi-GPU systems.
-   **Modular Design**: Each DRP model or group is encapsulated in its own Nextflow sub-workflow, enabling flexible execution.
-   **Integrated Voter**: A powerful module to create an ensemble prediction from the results of individual models.
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

This is the simplest way to get a complete result. The `run_ensemble.sh` script runs all model workflows in order and saves all results to a unique, timestamped directory.

-   **Run with the built-in test data:**
    ```bash
    ./run_ensemble.sh
    ```

-   **Run with your own data and custom GPU allocation:**
    You can pass any Nextflow parameter directly to the script. These will be automatically forwarded to all underlying pipeline runs.
    ```bash
    ./run_ensemble.sh --gene_exp_file /path/to/exp.csv \
                      --mutation_file /path/to/mut.csv \
                      --gpu_part1 0 --gpu_deepcdr 1
    ```
    All results will be stored in a new directory, e.g., `ensemble_results_20231027_153000/`.

### Method 2: Modular Execution (Advanced)

If you only want to run a specific model or the voter, call `main.nf` directly with the `--entry` flag. Results will be saved to the directory specified by `--output_dir` (default: `./results`).

-   **Run the `part1` model group:**
    ```bash
    nextflow run main.nf --entry part1 --gpu_part1 0
    ```

-   **Run the `DeepCDR` workflow:**
    ```bash
    nextflow run main.nf --entry deepcdr --gpu_deepcdr 1
    ```

-   **Run the Voter on existing results:**
    This should be run after one or more model workflows have completed and their `*_predictions.csv` files are present in the output directory.
    ```bash
    nextflow run main.nf --entry voter --output_dir ./ensemble_results_20231027_153000
    ```

## Workflow Parameters

Parameters can be specified on the command line (e.g., `--gpu_part1 1`).

### Workflow Control

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `entry` | **Required for modular runs**. Specifies the workflow to run. Options: `part1`, `deepaeg`, `deepcdr`, `dipk_graphdrp`, `voter`. | `''` |
| `output_dir` | The root directory where all prediction results will be saved. | `${projectDir}/results` |

### GPU Allocation

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `gpu_part1` | GPU ID for the `part1` model group (BANDRP, DeepTTA, etc.). | `0` |
| `gpu_deepaeg` | GPU ID for the `DeepAEG` model. | `0` |
| `gpu_deepcdr` | GPU ID for the `DeepCDR` model. | `0` |
| `gpu_dipk_graphdrp` | GPU ID for the `DIPK` & `GraphDRP` model group. | `0` |

### Core Input Files

| Parameter | Description | Default (points to test data) |
| :--- | :--- | :--- |
| `drug_smiles` | CSV file with drug SMILES strings. | `${projectDir}/test/drug_sample.csv` |
| `gene_exp_file` | Gene expression data file. | `${projectDir}/test/gene_sample.csv` |
| `mutation_file` | Gene mutation data file. | `${projectDir}/test/mu_sample.csv` |
| `cnv_file` | Copy Number Variation (CNV) data file. | `${projectDir}/test/cnv_sample.csv` |
| `all_in_one_deepaeg`| Integrated multi-omics file for DeepAEG. | `${projectDir}/test/al_sample.csv` |
| `microrna_file` | MicroRNA expression data file. | `${projectDir}/test/mi_sample.csv` |
| `gsva_file` | GSVA analysis results file. | `${projectDir}/test/gsva_sample.csv` |
| `drug_features_file`| Drug features file for GADRP and NeRD. | `${projectDir}/test/drug_with_conditions.csv` |
| `cell_file_graphdrp`| Cell line features file for GraphDRP. | `${projectDir}/test/mu_sample.csv` |

### Ensemble Voter Parameters

These flags (`1`=enable, `0`=disable) control which models are included in the final vote.

| Parameter | Default | Parameter | Default |
| :--- | :--- | :--- | :--- |
| `predict_lnic50` | `0` | `GPDRP_GIN` | `1` |
| `BANDRP` | `1` | `GPDRP_GINTransformer` | `1` |
| `DeepAEG` | `1` | `GraphDRP_GATNet` | `1` |
| `DeepCDR` | `1` | `GraphDRP_GAT_GCN` | `1` |
| `DeepTTA` | `1` | `GraphDRP_GCNNet` | `1` |
| `DIPK` | `1` | `GraphDRP_GINConvNet` | `1` |
| `GADRP` | `1` | `NERD` | `1` |
| `GPDRP_GAT` | `1` | `paccmann` | `1` |
| `GPDRP_GCN` | `1` | `Precily` | `1` |

## Project Structure
```
.
├── main.nf                 # Main Nextflow routing script
├── run_ensemble.sh         # One-click script for the full sequential workflow
├── nextflow.config         # (Optional) Nextflow configuration file
├── workflows/              # Directory for all sub-workflows
│   ├── nf_part1.nf
│   ├── nf_DeepAEG.nf
│   ├── nf_deepcdr.nf
│   ├── nf_DIPK-GraphDRP.nf
│   └── voter.nf
├── test/                   # Sample input data for testing
└── local_wheels/           # Local Python .whl packages
```

## Example Commands

**1. Run the full ensemble workflow using custom data and multiple GPUs:**
```bash
./run_ensemble.sh --drug_smiles /data/drugs.csv \
                  --gene_exp_file /data/expression.csv \
                  --gpu_part1 0 \
                  --gpu_deepcdr 1 \
                  --gpu_dipk_graphdrp 2
```

**2. Run only the `DIPK-GraphDRP` workflow on GPU 3:**
```bash
nextflow run main.nf --entry dipk_graphdrp --gpu_dipk_graphdrp 3
```

**3. Run the Voter on a specific results directory, excluding `DeepTTA` from the ensemble:**
```bash
nextflow run main.nf --entry voter \
                     --output_dir ./ensemble_results_20231027_153000 \
                     --DeepTTA 0
```
