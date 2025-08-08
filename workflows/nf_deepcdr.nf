#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    DeepCDR 子工作流 (V5 - Parameterized)
========================================================================================
*/

workflow run_deepcdr_workflow {
    take:
        gpu_id
        output_dir

    main:
        run_DeepCDR(gpu_id, output_dir)
}

process run_DeepCDR {
    tag "DeepCDR_Prediction"

    publishDir "${params.output_dir}", mode: 'copy', pattern: "DeepCDR_predictions.csv"

    input:
    val gpu_id
    val output_dir 

    output:
    path "DeepCDR_predictions.csv" 

    script:
    // 内置文件路径
    def model_file = "${projectDir}/DeepCDR/prog/saved_models/bd1.h5"
    def align_gexp_file = "${projectDir}/DeepCDR/depmap/CCLE/exp.csv"
    def align_mut_file = "${projectDir}/DeepCDR/depmap/CCLE/mu.csv"

    def output_file = "DeepCDR_predictions.csv"
    """
    echo "--- Starting DeepCDR Prediction on GPU ${gpu_id} ---"
    
    export CUDA_VISIBLE_DEVICES=${gpu_id}

    python ${projectDir}/DeepCDR/prog/predict_all.py \\
        --model_file "${model_file}" \\
        --drugs_file "${params.drug_smiles}" \\
        --mut_file "${params.mutation_file}" \\
        --gexp_file "${params.gene_exp_file}" \\
        --methy_file "${params.mutation_file}" \\
        --align_gexp_file "${align_gexp_file}" \\
        --align_mut_file "${align_mut_file}" \\
        --output_file "${output_file}"

    if [ ! -f "${output_file}" ]; then
        echo "ERROR: DeepCDR script failed. Output file was not created." >&2
        exit 1
    fi
    """
}