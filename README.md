# DRIVE: Drug Response Integration and Voting Ensemble

**DRIVE** is a comprehensive Nextflow-based computational pipeline designed for robust Drug Response Prediction (DRP). It integrates a curated ensemble of state-of-the-art deep learning models (including DIPK, GraphDRP, DeepTTC, BANDRP, etc.) to generate consensus predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Method 1: Docker (Recommended)](#method-1-docker-execution-easiest)
  - [Method 2: Smart Ensemble (For Large Datasets)](#method-2-smart-ensemble-workflow-for-large-scale-data)
  - [Method 3: Local Standard Execution](#method-3-local-standard-execution)
- [Workflow Parameters](#workflow-parameters)
  - [GPU Allocation](#gpu-allocation)
  - [Input Files](#input-files)

## Overview

Predicting cancer cell line response to drugs is computationally intensive. DRIVE orchestrates multiple complex models, managing their specific environment dependencies (Python 3.7/3.8, PyTorch versions) and hardware resources. 

The pipeline aggregates predictions from individual models using a `Voter` module (Ensemble Analysis) to produce a final, high-confidence prediction.

## Features

-   **Docker Support**: A fully encapsulated container image containing all model environments, eliminating local dependency hell.
-   **Smart Resource Planner**: The `smart_ensemble.sh` system automatically detects hardware (RAM/GPU). It chunks large datasets (e.g., 50x50 matrices for CPU-only modes or dynamic sizing for GPUs) to prevent memory overflows.
-   **Granular GPU Control**: Assign specific models to specific GPU IDs via `main.nf`.
-   **Automated Ensemble**: Merges predictions via a Random Forest-based voter or weighted averaging.

## Prerequisites

### For Docker Mode (Recommended)
-   **Docker Engine** installed.
-   **NVIDIA Container Toolkit** (for GPU acceleration).

### For Local Mode
-   **Nextflow** (v21.10+).
-   **Conda**: For environment management (auto-created from YAMLs).
-   **NVIDIA GPU**: Required for model inference.

## Project Structure

```text
DRIVE/
├── BANDRP-main/                # Model Source Codes
├── DeepTTC/
├── DIPK-main/
├── GPDRP/
├── GraphDRP-master/
├── paccmann_predictor-master/
├── Precily-v1.0.0/
├── environments/               # Conda environment YAMLs
├── local_wheels/               # Offline Python dependencies
├── results/                    # Default output directory
├── test/                       # Sample datasets
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

## Getting Started

### Method 1: Docker Execution (Easiest)

This method runs the full pipeline without any local installation.

#### 1. Pull the Image
```bash
docker pull crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3
```

#### 2. Run with Built-in Test Data
To check if everything works using the included sample data:
```bash
docker run --rm --gpus all \
  -v $(pwd)/final_check:/app/results \
  crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3
```

#### 3. Run with Your Own Data (Custom Input)
To use your own datasets, **mount your data folder** into the container (e.g., map local path to `/data`) and specify the paths using the standard omics flags.

**Example Command:**
Assuming your data files (`drug.csv`, `gene.csv`, etc.) are located in `/home/user/my_omics_data/`:

```bash
docker run --rm --gpus all \
  -v $(pwd)/my_results:/app/results \
  -v /home/user/my_omics_data:/data \
  crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3 \
  --drug_smiles "/data/drug.csv" \
  --gene_exp_file "/data/gene.csv" \
  --mutation_file "/data/mutation.csv" \
  --cnv_file "/data/cnv.csv" \
  --gsva_file "/data/gsva.csv"
```

### Method 2: Smart Ensemble Workflow (For Large-Scale Data)

Use this method if you have a large number of drugs/cells (e.g., >1000 pairs). The system automatically calculates available RAM/CPU and splits input data into "chunks" to run in parallel.

1.  **Configure Paths:** Edit `smart_ensemble.sh` to point to your input files.
2.  **Run:**
    ```bash
    chmod +x smart_ensemble.sh
    ./smart_ensemble.sh
    ```

### Method 3: Local Standard Execution

For standard-sized datasets on a machine with configured Conda/Nextflow.

#### 1. Setup
```bash
chmod +x run_ensemble.sh
```

#### 2. Run with Built-in Test Data
```bash
./run_ensemble.sh
```

#### 3. Run with Your Own Data (Custom Input)
Pass the absolute paths to your local files using the standard flags.

```bash
./run_ensemble.sh \
  --drug_smiles "/abs/path/to/drug.csv" \
  --gene_exp_file "/abs/path/to/gene.csv" \
  --mutation_file "/abs/path/to/mutation.csv" \
  --cnv_file "/abs/path/to/cnv.csv" \
  --gsva_file "/abs/path/to/gsva.csv"
```

## Workflow Parameters

These parameters apply to both Local (`run_ensemble.sh`) and Docker modes.

### GPU Allocation
You can control which GPU handles which model group in `main.nf`:

| Parameter | Default GPU | Description |
| :--- | :--- | :--- |
| `gpu_map.GPDRP` | 0 | GPDRP Model |
| `gpu_map.BANDRP` | 0 | BANDRP Model |
| `gpu_map.DeepTTC` | 0 | DeepTTC Model |
| `gpu_map.paccmann`| 0 | PaccMann Model |
| `gpu_map.Precily` | 0 | Precily Model |
| `gpu_map.DIPK` | 0 | DIPK Model |
| `gpu_map.GraphDRP`| 0 | GraphDRP Model |

### Input Files (Standardized Omics Names)
Override these flags to use your own data.

| Flag | Description | Expected Format |
| :--- | :--- | :--- |
| `--drug_smiles` | Drug Data | CSV: `drug_id,smiles` |
| `--gene_exp_file` | Gene Expression | CSV: Rows=Samples, Cols=Genes |
| `--mutation_file` | Mutation Data | CSV: Binary or frequency matrix |
| `--cnv_file` | Copy Number Variation | CSV: Copy number values |
| `--gsva_file` | GSVA Pathway Scores | CSV: Rows=Samples, Cols=Pathways |
| `--cell_file_graphdrp` | Cell Features | CSV: (Specific to GraphDRP, usually same as mutation) |