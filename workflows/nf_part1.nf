#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    SUBWORKFLOW MODULE (Part 1 Models) - V7.2 Explicit Dependencies
    - This version correctly accepts 4 input arguments.
========================================================================================
*/

workflow run_part1_workflow {
    take:
        gpu_id
        local_wheels_dir
        output_dir
        input_params // <-- 【【【 核心修正：声明接收第4个参数 'input_params' 】】】

    main:
        Channel
            .fromList([
                ['GAT', 'model_GAT_GDSC_drug_blind_run2.model'],
                ['GCN', 'model_GCN_GDSC_drug_blind_run2.model']
            ])
            .set { gpdrp_models_ch }

        run_GPDRP(
            gpu_id, 
            local_wheels_dir, 
            output_dir,
            gpdrp_models_ch.map { it[0] },
            gpdrp_models_ch.map { it[1] },
            input_params // Pass the map to the process
        )
        
        run_BANDRP(gpu_id, local_wheels_dir, output_dir, input_params)
        run_DeepTTC(gpu_id, local_wheels_dir, output_dir, input_params)
        run_paccmann(gpu_id, local_wheels_dir, output_dir, input_params)
        run_Precily(gpu_id, local_wheels_dir, output_dir, input_params)
}

process run_BANDRP {
    tag "BANDRP_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir; val p // Receive the map as 'p'
    script:
    def target_output = "${output_dir}/bandrp_predictions_part_1.csv"
    """
    mkdir -p ${output_dir}
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/BANDRP-main
    python ./predict_all.py --model_path ./github_upload/output_dir/db1/model.pt --new_drugs_csv "${p.drug_smiles}" --exp_path "${p.gene_exp_file}" --mut_path "${p.mutation_file}" --cnv_path "${p.cnv_file}" --output_csv "${target_output}" --drug_batch_size 100000 --cuda_id ${gpu_id}
    """
}

process run_DeepTTC {
    tag "DeepTTC_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir; val p
    script:
    def target_output = "${output_dir}/DeepTTC_predictions.csv"
    """
    mkdir -p ${output_dir}
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/DeepTTC
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --new_drug_file "${p.drug_smiles}" --new_cell_line_file "${p.gene_exp_file}" --training_gene_list_file "./mydata/expt.txt" --model_dir "./DeepTTC" --vocab_dir "./ESPF" --output_file "${target_output}"
    """
}

process run_GPDRP {
    tag "GPDRP predict [${model_type}]"
    input:
    val gpu_id; path local_wheels_dir; val output_dir; val model_type; val model_filename; val p
    script:
    def target_output = "${output_dir}/GPDRP_predictions_${model_type}.csv"
    """
    mkdir -p ${output_dir}
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/GPDRP
    python ./predict_all.py --smiles_file "${p.drug_smiles}" --new_cell_line_file "${p.gsva_file}" --training_gene_expression_file ./mydata/exp.txt --model_file "./output/models/${model_filename}" --output_file "${target_output}" --model_type "${model_type}" --cuda_name "cuda:${gpu_id}"
    """
}

process run_paccmann {
    tag "paccmann_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir; val p
    script:
    def target_output = "${output_dir}/paccmann_predictions.csv"
    """
    mkdir -p ${output_dir}
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/paccmann_predictor-master
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --predict_smiles_filepath "${p.drug_smiles}" --gep_filepath "${p.gene_exp_file}" --model_run_path ./paccman_training_runs/paccmann_train_1747977832 --gene_filepath_spec ./data/2128_genes.pkl --smiles_language_filepath ./paccmann/smiles_language.pkl --output_filepath "${target_output}"
    """
}

process run_Precily {
    tag "Precily_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir; val p
    script:
    def target_output = "${output_dir}/Precily_predictions.csv"
    """
    mkdir -p ${output_dir}
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/Precily-v1.0.0/Pathway_based
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --input_drugs_file "${p.drug_smiles}" --input_cell_lines_file "${p.gsva_file}" --models_dir . --output_file "${target_output}" --drug_embedding_file ../mydata/utils/drug.pubchem.canon.l8.ws20.txt --elements_file ../mydata/utils/elements.txt --cell_features_template_file ../mydata/mycell_gsva2.csv
    """
}