# Minimal toy example for running DRIVE

This page provides a minimal toy example for running the DRIVE workflow using the small test files distributed with the repository. The purpose of this example is to help users verify that the workflow can be executed correctly, rather than to reproduce the full-scale benchmarking or generate biologically meaningful drug-response conclusions.

The toy example covers the complete DRIVE execution path, including input loading, model-level prediction, ensemble integration, and final result generation.

## Purpose of the toy example

The toy dataset is designed for quick workflow validation. It allows users to check whether:

- the required input files are correctly formatted;
- sample and drug identifiers can be recognized by the workflow;
- molecular feature matrices can be matched across modalities;
- individual model prediction modules can be launched;
- the DRIVE ensemble module can integrate model-level outputs;
- the expected final prediction file is generated.

Because the dataset is intentionally small, the output should only be used to confirm successful execution. It should not be interpreted as a formal drug-response prediction result.

## Toy dataset overview

The toy data are located in the `test/` directory of the repository. They include a small drug file and several matched molecular feature matrices.

```text
test/
├── drug_sample.csv
├── gene_sample.csv
├── mu_sample.csv
├── cnv_sample.csv
└── gsva_sample.csv
```

In DRIVE, the input files include drug structure information and sample-level molecular features. The molecular features include gene expression, mutation, copy number variation, and GSEA pathway score features.

## Input file requirements

| File name | Required | Data type | Description | Expected format |
|---|---:|---|---|---|
| `drug_sample.csv` | Yes | Drug structure | Drug identifiers and SMILES strings | CSV with columns `drug_id` and `smiles` |
| `gene_sample.csv` | Yes | Gene expression | Sample-level gene expression features | Rows represent samples; columns represent genes |
| `mu_sample.csv` | Yes | Mutation | Sample-level mutation features | Rows represent samples; columns represent genes |
| `cnv_sample.csv` | Yes | Copy number variation | Sample-level CNV features | Rows represent samples; columns represent genes |
| `gsva_sample.csv` | Yes | GSEA pathway score | Sample-level pathway score features | Rows represent samples; columns represent pathways or gene sets |

> Note: the workflow parameter is named `--gsva_file` for compatibility with the existing implementation, but the current toy input is used as a GSEA pathway score matrix.

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

### Sample-level molecular feature matrices

Most molecular input files in DRIVE follow the same sample-by-feature structure. This includes `gene_sample.csv`, `mu_sample.csv`, and `cnv_sample.csv`.

```csv
sample_id,FEATURE1,FEATURE2,FEATURE3
Sample_001,0.23,1.45,0.00
Sample_002,0.11,0.98,1.20
```

In these files:

- rows represent samples or cell lines;
- columns represent molecular features;
- the first column should contain sample identifiers;
- feature names should be placed in the header row;
- sample identifiers should be consistent across all required molecular feature files.

The exact meaning of the feature values depends on the data type. For example, gene expression files contain expression values, mutation files may contain binary mutation indicators or preprocessed mutation features, and CNV files contain copy number variation features.

### GSEA pathway score matrix

The pathway-level input file is represented by `gsva_sample.csv` in the toy example. In the current workflow configuration, this file is used as a GSEA pathway score matrix.

```csv
sample_id,HALLMARK_APOPTOSIS,HALLMARK_GLYCOLYSIS,KEGG_CELL_CYCLE
Sample_001,0.42,-0.18,0.77
Sample_002,0.15,0.31,-0.22
```

Rows should represent samples or cell lines, and columns should represent pathways, gene sets, or curated signatures. The GSEA scoring strategy should be kept consistent within a single run.

### Other optional or model-specific feature files

Some component models may support additional feature representations. These files are not required for the minimal toy run unless the corresponding model configuration explicitly depends on them.

When optional or custom feature files are used, they should follow the same general convention:

- rows represent samples, cell lines, drugs, or drug-sample pairs depending on the model requirement;
- columns represent features;
- identifiers should be consistent with the corresponding drug or sample IDs used elsewhere in the workflow;
- missing values should be handled before running DRIVE unless the relevant model explicitly supports them.

## Identifier consistency

All required sample-level molecular feature files should use consistent sample identifiers. For example, if `Sample_001` appears in `gene_sample.csv`, the same identifier should also appear in `mu_sample.csv`, `cnv_sample.csv`, and `gsva_sample.csv`.

Inconsistent sample identifiers are one of the most common causes of failed or incomplete runs. Before running DRIVE on custom data, users are encouraged to check that the sample IDs overlap across all required molecular feature files.

The drug identifiers in `drug_sample.csv` should also be stable and unique, because they are used to label model-level and ensemble-level prediction outputs.

## Run the toy example locally

The recommended local command is to use the wrapper script:

```bash
chmod +x run_ensemble.sh
./run_ensemble.sh
```

The wrapper script runs the three workflow stages in order:

1. DIPK and GraphDRP prediction
2. Part 1 model prediction
3. Final ensemble voting

To explicitly pass the toy input files, run:

```bash
./run_ensemble.sh \
  --drug_smiles "test/drug_sample.csv" \
  --gene_exp_file "test/gene_sample.csv" \
  --mutation_file "test/mu_sample.csv" \
  --cnv_file "test/cnv_sample.csv" \
  --gsva_file "test/gsva_sample.csv" \
  --cell_file_graphdrp "test/mu_sample.csv"
```

The local wrapper writes outputs to `results/` by default.

## Run the toy example with Docker

The Docker image contains the built-in toy input files. To run the toy example and write results to a local output folder, use:

```bash
docker run --rm --gpus all \
  -v $(pwd)/toy_results:/app/results \
  crpi-c4pny7ppnuyy2551.cn-hangzhou.personal.cr.aliyuncs.com/ldqq/ldqq001:v3
```

The Docker workflow writes outputs inside `/app/results`, which is mounted to `toy_results/` in the command above.

## Expected outputs

The previous toy example used a generic `base_predictions/`, `ensemble_predictions/`, and `logs/` layout. The current workflow writes model-level prediction files directly into the output directory.

### Local run

After a successful local run, the output directory should look similar to:

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

### Docker run

If Docker is used with the command above, the corresponding files will be written to the mounted output directory:

```text
toy_results/
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
|---|---|
| `input_files/prepared_drug_list.csv` | Drug list prepared by the workflow before model prediction |
| `*_predictions*.csv` | Model-level prediction files generated by individual DRP models |
| `final_results.tsv` | Final consensus drug response prediction file generated by the ensemble module |

The exact set of model-level prediction files may vary depending on the selected workflow entry point and model configuration.

## How to check whether the run was successful

A minimal run can be considered successful if:

1. the workflow finishes without fatal errors;
2. model-level prediction files are generated in the output directory;
3. `final_results.tsv` is present in the output directory;
4. the final output file contains drug-response prediction results.

If the workflow stops unexpectedly, first check whether the input files exist, whether the sample identifiers are consistent across feature matrices, and whether Docker or Nextflow can access the required GPU environment.

## Notes and limitations

1. The toy dataset is only intended for workflow testing.
2. The small input size may not reflect the runtime, memory usage, or predictive performance of full-scale datasets.
3. All sample-level molecular feature files should use consistent sample identifiers.
4. The drug input file must contain valid SMILES strings for all compounds.
5. The pathway-level input currently represents GSEA pathway score features, although the parameter name remains `--gsva_file`.
6. Optional feature files should be added only when the selected model configuration requires them.
7. For large-scale datasets, users may alternatively run `smart_ensemble.sh`, which enables automatic chunking and resource-aware scheduling.
