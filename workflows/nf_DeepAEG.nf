#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    MODULAR WORKFLOW (DeepAEG Only) - V5
========================================================================================
*/

process run_DeepAEG {
    tag "DeepAEG_predict"
    
    input:
    val gpu_id
    val output_dir

    script:
    def output_filepath = "${output_dir}/DeepAEG_predictions.csv"
    """
    mkdir -p ${output_dir}
    
    cd ${projectDir}/DeepAEG-main/prog/
    
    python ./predict_all.py \\
        -model_path ./MyBestDeepAEG_0.7789226722858869.h5 \\
        -new_drug_file ${params.drug_smiles} \\
        -gene_info_file ${params.all_in_one_deepaeg} \\
        -output_file "${output_filepath}" \\
        -vocab_path_bpe ./ESPF/drug_codes_chembl_freq_1500.txt \\
        -subword_csv_path_bpe ./ESPF/subword_units_map_chembl_freq_1500.csv \\
        -gpu_id ${gpu_id}

    if [ ! -f "${output_filepath}" ]; then
        echo "ERROR: DeepAEG script finished, but the output file was NOT created." >&2
        exit 1
    fi
    """
}