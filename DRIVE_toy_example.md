# Minimal toy example for running DRIVE

This page provides a minimal toy example for running the DRIVE workflow using the small test files distributed with the repository. The purpose of this example is to help users verify that the workflow can be executed correctly, rather than to reproduce the full-scale benchmarking or generate biologically meaningful drug-response conclusions.

The toy example covers the complete DRIVE execution path, including input loading, model-level prediction, ensemble integration, and final result generation.

## Purpose of the toy example

The toy dataset is designed for quick workflow validation. It allows users to check whether:

- the required input files are correctly formatted;
- sample and drug identifiers can be recognized by the workflow;
- individual model prediction modules can be launched;
- the DRIVE ensemble module can integrate model-level outputs;
- the expected output directory and final prediction file are generated.

Because the dataset is intentionally small, the output should only be used to confirm successful execution. It should not be interpreted as a formal drug-response prediction result.

## Toy dataset overview

The toy data are located in the `test/` directory of the repository. They include a small drug file and several matched omics matrices.

```text
test/
├── drug_sample.csv
├── gene_sample.csv
├── mu_sample.csv
├── cnv_sample.csv
├── gsva_sample.csv
└── mi_sample.csv
```

Each file represents one input modality used by DRIVE. The drug file provides compound identifiers and SMILES strings, while the omics files provide molecular features for the same set of samples or cell lines.

## Input file requirements

| File name | Required | Description | Expected format |
|---|---:|---|---|
| `drug_sample.csv` | Yes | Drug identifiers and SMILES strings | CSV with columns `drug_id` and `smiles` |
| `gene_sample.csv` | Yes | Gene expression matrix | Rows represent samples; columns represent genes |
| `mu_sample.csv` | Yes | Mutation matrix | Rows represent samples; columns represent genes |
| `cnv_sample.csv` | Yes | Copy number variation matrix | Rows represent samples; columns represent genes |
| `gsva_sample.csv` | Yes | GSVA pathway score matrix | Rows represent samples; columns represent pathways |
| `mi_sample.csv` | No | MicroRNA expression matrix | Rows represent samples; columns represent miRNAs |

## Data format notes

### Drug file

The drug file should contain at least two columns:

| Column | Description |
|---|---|
| `drug_id` | Unique drug identifier used by the workflow |
| `smiles` | Canonical or valid SMILES string for the compound |

Example structure:

```csv
drug_id,smiles
Drug_001,CCOC(=O)C1=CC=CC=C1
Drug_002,CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

The `drug_id` values are used to label prediction outputs, while the `smiles` column is used by models that require molecular structure information.

### Omics matrices

The omics files should be comma-separated matrices. In each matrix:

- rows represent samples or cell lines;
- columns represent molecular features;
- row identifiers should be consistent across omics files;
- feature names should be placed in the header row.

Example structure:

```csv
sample_id,GENE1,GENE2,GENE3
Sample_001,0.23,1.45,0.00
Sample_002,0.11,0.98,1.20
```

For mutation data, values are typically binary or mutation-frequency-like features. For CNV data, values represent copy number variation features. For GSVA data, columns represent pathway-level scores rather than individual genes.

## Identifier consistency

All required omics matrices should use consistent sample identifiers. For example, if `Sample_001` appears in `gene_sample.csv`, the same identifier should also appear in `mu_sample.csv`, `cnv_sample.csv`, and `gsva_sample.csv`.

Inconsistent sample identifiers are one of the most common causes of failed or incomplete runs. Before running DRIVE on custom data, users are encouraged to check that the sample IDs overlap across all required omics files.

The microRNA file is optional. If microRNA data are unavailable, the workflow can be run without `mi_sample.csv`, provided that the corresponding parameter is omitted or handled according to the workflow settings.

## Run the toy example with Nextflow

From the repository root directory, run:

```bash
nextflow run main.nf \
  --drug_smiles "test/drug_sample.csv" \
  --gene_exp_file "test/gene_sample.csv" \
  --mutation_file "test/mu_sample.csv" \
  --cnv_file "test/cnv_sample.csv" \
  --gsva_file "test/gsva_sample.csv" \
  --microrna_file "test/mi_sample.csv" \
  -resume
```

## Run the toy example with Docker

The following command mounts the local `test/` directory as input and writes workflow outputs to `toy_results/`:

```bash
docker run --rm --gpus all \
  -v $(pwd)/test:/data \
  -v $(pwd)/toy_results:/app/results \
  crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3 \
  --drug_smiles "/data/drug_sample.csv" \
  --gene_exp_file "/data/gene_sample.csv" \
  --mutation_file "/data/mu_sample.csv" \
  --cnv_file "/data/cnv_sample.csv" \
  --gsva_file "/data/gsva_sample.csv" \
  --microrna_file "/data/mi_sample.csv"
```

## Expected outputs

After a successful run, the output directory should contain files similar to the following:

| Output path | Description |
|---|---|
| `results/base_predictions/` | Prediction outputs generated by individual base models |
| `results/ensemble_predictions/` | Integrated prediction outputs generated by the DRIVE ensemble module |
| `results/logs/` | Workflow execution logs and model-level runtime records |
| `results/final_results.tsv` | Final consensus drug response prediction file |

If Docker is used with the example command above, the corresponding files will be written to the mounted output directory:

```text
toy_results/
├── base_predictions/
├── ensemble_predictions/
├── logs/
└── final_results.tsv
```

## How to check whether the run was successful

A minimal run can be considered successful if:

1. the workflow finishes without fatal errors;
2. model-level prediction files are generated under `base_predictions/`;
3. ensemble prediction files are generated under `ensemble_predictions/`;
4. `final_results.tsv` is present in the output directory;
5. the log files are available under `logs/`.

Users should first inspect `results/logs/` if the workflow stops unexpectedly.

## Notes and limitations

1. The toy dataset is only intended for workflow testing.
2. The small input size may not reflect the runtime, memory usage, or predictive performance of full-scale datasets.
3. All omics matrices should use consistent sample identifiers.
4. The drug input file must contain valid SMILES strings for all compounds.
5. The microRNA input file is optional and may be omitted if not available.
6. For large-scale datasets, users may alternatively run `smart_ensemble.sh`, which enables automatic chunking and resource-aware scheduling.
