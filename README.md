# DRIVE: Drug Response Integration and Voting Ensemble

**DRIVE** is a Nextflow-based computational pipeline designed for robust Drug Response Prediction (DRP). It integrates a curated ensemble of published deep learning-based DRP models, including DIPK, GraphDRP, DeepTTC, BANDRP, and related models, to generate consensus predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick test with toy data](#quick-test-with-toy-data)
- [Getting Started](#getting-started)
  - [Method 1: Docker Execution](#method-1-docker-execution)
  - [Method 2: Smart Ensemble Workflow](#method-2-smart-ensemble-workflow)
  - [Method 3: Local Standard Execution](#method-3-local-standard-execution)
- [Expected outputs](#expected-outputs)
- [Workflow Parameters](#workflow-parameters)
  - [GPU Allocation](#gpu-allocation)
  - [Input Files](#input-files)
- [Troubleshooting](#troubleshooting)

## Overview

Predicting cancer cell line response to drugs is computationally intensive. DRIVE orchestrates multiple model modules while managing their environment dependencies, runtime requirements, and hardware resources.

The pipeline aggregates predictions from individual models using an ensemble module to produce a final consensus drug response prediction.

## Features

- **Docker Support**: A containerized execution mode containing the required model environments, reducing local dependency conflicts.
- **Smart Resource Planner**: The `smart_ensemble.sh` system automatically detects hardware resources and chunks large datasets to reduce the risk of memory overflows.
- **Granular GPU Control**: Specific models can be assigned to specific GPU IDs via `main.nf`.
- **Automated Ensemble**: Predictions from individual models are merged through the DRIVE ensemble module to produce final consensus outputs.

## Prerequisites

### For Docker Mode

- **Docker Engine** installed.
- **NVIDIA Container Toolkit** for GPU acceleration.

### For Local Mode

- **Nextflow** (v21.10+).
- **Conda** for environment management.
- **NVIDIA GPU** for model inference.

## Project Structure

```text
DRIVE/
├── BANDRP-main/                # Model source codes
├── DeepTTC/
├── DIPK-main/
├── GPDRP/
├── GraphDRP-master/
├── paccmann_predictor-master/
├── Precily-v1.0.0/
├── environments/               # Conda environment YAMLs
├── local_wheels/               # Offline Python dependencies
├── results/                    # Default output directory
├── test/                       # Toy datasets
├── workflows/                  # Sub-workflow definitions (.nf)
├── Dockerfile                  # Container definition
├── main.nf                     # Local execution entry point
├── main_docker.nf              # Docker execution entry point
├── run_ensemble.sh             # Standard local run script
├── run_ensemble_dockfile.sh    # Internal Docker entry script
├── smart.nf                    # Resource planning workflow
├── smart_ensemble.sh           # Smart workflow entry script
└── voter.py                    # Ensemble aggregation logic
```

## Quick test with toy data

A minimal toy dataset is provided in the `test/` directory to help users verify that DRIVE is correctly installed and executable.

For detailed input format descriptions and expected outputs, see [`docs/minimal_toy_example.md`](docs/minimal_toy_example.md).

## Getting Started

### Method 1: Docker Execution

This method runs the full pipeline without manually configuring local model environments.

#### 1. Pull the image

```bash
docker pull crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3
```

If the pre-built image cannot be pulled from the registry, users can build the image locally using the provided `Dockerfile`:

```bash
docker build -t drive:latest .
```

#### 2. Run with built-in test data

To check whether the workflow can run using the included toy data:

```bash
docker run --rm --gpus all \
  -v $(pwd)/final_check:/app/results \
  crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3
```

#### 3. Run with custom input data

To use your own datasets, mount your data folder into the container and specify the paths using the standard input flags.

Example command assuming your data files are located in `/home/user/my_data/`:

```bash
docker run --rm --gpus all \
  -v $(pwd)/my_results:/app/results \
  -v /home/user/my_data:/data \
  crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3 \
  --drug_smiles "/data/drug.csv" \
  --gene_exp_file "/data/gene.csv" \
  --mutation_file "/data/mutation.csv" \
  --cnv_file "/data/cnv.csv" \
  --gsva_file "/data/gsea.csv"
```

### Method 2: Smart Ensemble Workflow

Use this method for large-scale datasets with many drug-cell pairs. The workflow calculates available hardware resources and splits the input data into chunks for more stable execution.

1. Configure paths in `smart_ensemble.sh` to point to your input files.
2. Run:

```bash
chmod +x smart_ensemble.sh
./smart_ensemble.sh
```

### Method 3: Local Standard Execution

Use this method for standard-sized datasets on a machine with configured Conda and Nextflow environments.

#### 1. Setup

```bash
chmod +x run_ensemble.sh
```

#### 2. Run with built-in test data

```bash
./run_ensemble.sh
```

#### 3. Run with custom input data

Pass the paths to your local files using the standard input flags.

```bash
./run_ensemble.sh \
  --drug_smiles "/path/to/drug.csv" \
  --gene_exp_file "/path/to/gene.csv" \
  --mutation_file "/path/to/mutation.csv" \
  --cnv_file "/path/to/cnv.csv" \
  --gsva_file "/path/to/gsea.csv"
```

## Expected outputs

After a successful run, DRIVE generates model-level predictions, ensemble-level predictions, execution logs, and the final consensus prediction file.

| Output path | Description |
| :--- | :--- |
| `results/base_predictions/` | Prediction outputs generated by individual base models |
| `results/ensemble_predictions/` | Integrated prediction outputs generated by the DRIVE ensemble module |
| `results/logs/` | Workflow execution logs and model-level runtime records |
| `results/final_results.tsv` | Final consensus drug response prediction file |

If Docker is used with a mounted output directory, the same output files will be written to the mounted results folder.

## Workflow Parameters

These parameters apply to both local and Docker modes.

### GPU Allocation

You can control which GPU handles which model group in `main.nf`:

| Parameter | Default GPU | Description |
| :--- | :--- | :--- |
| `gpu_map.GPDRP` | 0 | GPDRP model |
| `gpu_map.BANDRP` | 0 | BANDRP model |
| `gpu_map.DeepTTC` | 0 | DeepTTC model |
| `gpu_map.paccmann` | 0 | PaccMann model |
| `gpu_map.Precily` | 0 | Precily model |
| `gpu_map.DIPK` | 0 | DIPK model |
| `gpu_map.GraphDRP` | 0 | GraphDRP model |

### Input Files

Override these flags to use your own data.

| Flag | Description | Expected format |
| :--- | :--- | :--- |
| `--drug_smiles` | Drug identifiers and SMILES strings | CSV with columns `drug_id` and `smiles` |
| `--gene_exp_file` | Gene expression features | CSV: rows = samples, columns = genes |
| `--mutation_file` | Mutation features | CSV: rows = samples, columns = genes |
| `--cnv_file` | Copy number variation features | CSV: rows = samples, columns = genes |
| `--gsva_file` | GSEA pathway scores | CSV: rows = samples, columns = pathways or gene sets |
| `--cell_file_graphdrp` | Cell features used by GraphDRP | CSV: model-specific cell feature matrix, usually aligned with mutation features |

Although the parameter name is `--gsva_file`, this input currently refers to GSEA pathway scores in the provided workflow configuration.

## Troubleshooting

- If Docker cannot access the GPU, check whether NVIDIA Container Toolkit is correctly installed.
- If the workflow stops because of missing files, confirm that all input paths are correctly mounted into the container.
- If sample-level outputs are incomplete, check whether sample identifiers are consistent across all molecular feature files.
- If a large-scale run exceeds memory limits, use `smart_ensemble.sh` for automatic chunking and resource-aware scheduling.
- If the Docker image cannot be pulled from the registry, build the image locally using the provided `Dockerfile`.
