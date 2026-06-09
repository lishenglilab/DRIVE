# DRIVE: Drug Response Integration and Voting Ensemble

**DRIVE** is a Nextflow-based computational pipeline designed for robust Drug Response Prediction (DRP). It integrates a curated ensemble of published deep learning-based DRP models, including DIPK, GraphDRP, DeepTTC, BANDRP, GPDRP, paccmann, and Precily, to generate consensus predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick test with toy data](#quick-test-with-toy-data)
- [Getting Started](#getting-started)
  - [Method 1: Docker Execution](#method-1-docker-execution)
  - [Method 2: Local Standard Execution](#method-2-local-standard-execution)
  - [Method 3: Smart Ensemble Workflow](#method-3-smart-ensemble-workflow)
- [Expected outputs](#expected-outputs)
  - [Standard local or Docker run](#standard-local-or-docker-run)
  - [Smart ensemble run](#smart-ensemble-run)
- [Workflow Parameters](#workflow-parameters)
  - [Workflow entry points](#workflow-entry-points)
  - [GPU Allocation](#gpu-allocation)
  - [Input Files](#input-files)
- [Troubleshooting](#troubleshooting)

## Overview

Predicting cancer cell line response to drugs is computationally intensive. DRIVE orchestrates multiple model modules while managing their environment dependencies, runtime requirements, and hardware resources.

The workflow is organized into three major stages:

1. **DIPK and GraphDRP prediction**
2. **Part 1 model prediction**, including BANDRP, DeepTTC, GPDRP, paccmann, and Precily
3. **Final ensemble voting**

The pipeline aggregates predictions from individual models using the DRIVE ensemble module to produce a final consensus drug response prediction.

## Features

- **Docker support**: A containerized execution mode containing the required model environments, reducing local dependency conflicts.
- **Local Nextflow execution**: A standard local mode for users with configured Conda and Nextflow environments.
- **Smart resource planner**: The `smart_ensemble.sh` workflow splits large input drug lists into chunks and schedules chunk-level jobs according to available hardware resources.
- **Granular GPU control**: Specific models can be assigned to specific GPU IDs through `main.nf`.
- **Automated ensemble**: Predictions from individual models are merged through the DRIVE ensemble module to produce final consensus outputs.

## Prerequisites

### For Docker mode

- **Docker Engine**
- **NVIDIA Container Toolkit** for GPU acceleration

### For local mode

- **Nextflow** (v21.10+)
- **Conda** for environment management
- **NVIDIA GPU** for model inference

## Project Structure

```text
DRIVE/
├── BANDRP-main/                 # BANDRP model source code
├── DeepTTC/                     # DeepTTC model source code
├── DIPK-main/                   # DIPK model source code
├── GPDRP/                       # GPDRP model source code
├── GraphDRP-master/             # GraphDRP model source code
├── paccmann_predictor-master/   # paccmann model source code
├── Precily-v1.0.0/              # Precily model source code
├── environments/                # Conda environment YAML files
├── local_wheels/                # Offline Python dependencies
├── results/                     # Default output directory for standard runs
├── results_final/               # Default final output directory for smart runs
├── smart_workspace/             # Intermediate workspace for smart runs
├── test/                        # Toy datasets
├── workflows/                   # Sub-workflow definitions
├── Dockerfile                   # Container definition
├── main.nf                      # Local execution entry point
├── main_docker.nf               # Docker execution entry point
├── nextflow.config              # Nextflow profile configuration
├── run_ensemble.sh              # Standard local run script
├── run_ensemble_dockerfile.sh   # Docker internal run script
├── smart.nf                     # Resource planning workflow
├── smart_ensemble.sh            # Smart workflow entry script
└── voter.py                     # Ensemble aggregation script
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

The results will be written to the mounted output directory, here `final_check/`.

#### 3. Run with custom input data

To use custom input data in Docker mode, mount your data folder into the container and pass the corresponding input paths.

```bash
docker run --rm --gpus all \
  -v $(pwd)/my_results:/app/results \
  -v /path/to/my_data:/data \
  crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3 \
  --drug_smiles "/data/drug.csv" \
  --gene_exp_file "/data/gene.csv" \
  --mutation_file "/data/mutation.csv" \
  --cnv_file "/data/cnv.csv" \
  --gsva_file "/data/gsea.csv" \
  --cell_file_graphdrp "/data/mutation.csv"
```

> Note: if your Docker image uses `run_ensemble_dockerfile.sh` as the entry script, make sure that command-line arguments are forwarded to the internal Nextflow calls. Otherwise, the container will run with the default paths inside `/app/test/`.

### Method 2: Local Standard Execution

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
  --gsva_file "/path/to/gsea.csv" \
  --cell_file_graphdrp "/path/to/mutation.csv"
```

By default, the standard local run writes outputs to `results/`.

### Method 3: Smart Ensemble Workflow

Use this method for large-scale datasets with many drug-cell pairs. The smart workflow first estimates hardware resources, splits the input drug list into chunks, runs model prediction for each chunk, consolidates chunk-level prediction files, and then runs the final ensemble stage.

#### 1. Configure input paths

Edit the input path variables in `smart_ensemble.sh`, including:

```bash
DRUG_SMILES="${BASE_DIR}/test/drug_sample.csv"
GENE_EXP="${BASE_DIR}/test/gene_sample.csv"
MUTATION="${BASE_DIR}/test/mu_sample.csv"
CNV="${BASE_DIR}/test/cnv_sample.csv"
GSVA="${BASE_DIR}/test/gsva_sample.csv"
CELL_GRAPHDRP="${BASE_DIR}/test/mu_sample.csv"
```

#### 2. Run

```bash
chmod +x smart_ensemble.sh
./smart_ensemble.sh
```

By default, the smart workflow writes final consolidated outputs to `results_final/` and intermediate chunk-level outputs to `smart_workspace/`.

## Expected outputs

### Standard local or Docker run

For standard local execution, the default output directory is:

```text
results/
```

For Docker execution with a mounted output directory, files are written to the host directory mounted to `/app/results`.

A successful standard run should generate files similar to the following:

```text
results/
├── input_files/
│   └── prepared_drug_list.csv
├── bandrp_predictions.csv
├── DeepTTC_predictions.csv
├── GPDRP_predictions_GAT.csv
├── GPDRP_predictions_GCN.csv
├── paccmann_predictions.csv
├── Precily_predictions.csv
├── DIPK_predictions.csv
├── GraphDRP_predictions_GATNet.csv
├── GraphDRP_predictions_GAT_GCN.csv
└── final_results.tsv
```

| Output file or directory | Description |
| :--- | :--- |
| `input_files/prepared_drug_list.csv` | Drug list prepared by the workflow before model prediction |
| `*_predictions*.csv` | Model-level prediction files generated by individual DRP models |
| `final_results.tsv` | Final consensus drug response prediction file generated by the ensemble module |

The exact set of model-level prediction files may vary depending on the selected workflow entry point and model configuration.

> Note: the convenience scripts clean previous output directories before running. Back up important results before starting a new run.

### Smart ensemble run

For smart large-scale execution, the default directories are:

```text
smart_workspace/
results_final/
```

A successful smart run should generate files similar to the following:

```text
smart_workspace/
├── plan/
│   ├── chunk_000.csv
│   ├── chunk_001.csv
│   ├── config.sh
│   └── resource_report.txt
└── results/
    ├── dipk_graphdrp_000/
    ├── dipk_graphdrp_001/
    ├── part1_000/
    └── part1_001/

results_final/
├── bandrp_predictions.csv
├── DeepTTC_predictions.csv
├── GPDRP_predictions_GAT.csv
├── GPDRP_predictions_GCN.csv
├── paccmann_predictions.csv
├── Precily_predictions.csv
├── DIPK_predictions.csv
├── GraphDRP_predictions_GATNet.csv
├── GraphDRP_predictions_GAT_GCN.csv
└── final_results.tsv
```

| Output file or directory | Description |
| :--- | :--- |
| `smart_workspace/plan/chunk_*.csv` | Chunked drug input files generated by the smart planner |
| `smart_workspace/plan/config.sh` | Automatically generated execution configuration, including chunk number and parallelism |
| `smart_workspace/plan/resource_report.txt` | Hardware and chunking summary generated by the planner |
| `smart_workspace/results/<entry>_<chunk_id>/` | Intermediate model prediction outputs for each chunk |
| `results_final/*_predictions*.csv` | Consolidated model-level prediction files after merging chunks |
| `results_final/final_results.tsv` | Final consensus prediction file generated by the ensemble module |

## Workflow Parameters

These parameters apply to local and Docker modes.

### Workflow entry points

The workflow uses three entry points internally:

| Entry point | Description |
| :--- | :--- |
| `dipk_graphdrp` | Runs DIPK and GraphDRP prediction modules |
| `part1` | Runs BANDRP, DeepTTC, GPDRP, paccmann, and Precily prediction modules |
| `voter` | Runs final ensemble voting on available model-level prediction files |

The provided wrapper scripts run these stages in order.

### GPU Allocation

You can control which GPU handles which model group in `main.nf`:

| Parameter | Default GPU | Description |
| :--- | :--- | :--- |
| `gpu_map.GPDRP` | 0 | GPDRP model |
| `gpu_map.BANDRP` | 0 | BANDRP model |
| `gpu_map.DeepTTC` | 0 | DeepTTC model |
| `gpu_map.paccmann` | 0 | paccmann model |
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
| `--gsva_file` | GSEA pathway score features | CSV: rows = samples, columns = pathways or gene sets |
| `--cell_file_graphdrp` | Cell features used by GraphDRP | CSV: model-specific cell feature matrix, usually aligned with mutation features |
| `--microrna_file` | Optional microRNA expression file | CSV: rows = samples, columns = miRNAs |

Although the parameter name is `--gsva_file`, this input currently refers to GSEA pathway score features in the provided workflow configuration.

## Troubleshooting

- If Docker cannot access the GPU, check whether NVIDIA Container Toolkit is correctly installed.
- If the workflow stops because of missing files, confirm that all input paths are correctly mounted into the container.
- If Docker custom input flags do not take effect, check whether the Docker entry script forwards command-line arguments to the internal Nextflow calls.
- If sample-level outputs are incomplete, check whether sample identifiers are consistent across all molecular feature files.
- If model-level prediction files are missing, check whether the corresponding workflow entry point was executed successfully.
- If a large-scale run exceeds memory limits, use `smart_ensemble.sh` for automatic chunking and resource-aware scheduling.
- If the Docker image cannot be pulled from the registry, build the image locally using the provided `Dockerfile`.
