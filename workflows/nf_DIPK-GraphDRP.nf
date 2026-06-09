#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    DIPK & GraphDRP SUB-WORKFLOW
========================================================================================
*/

process setup_dipk_graphdrp_env {
    tag "Setup DIPK/GraphDRP Env (Conda & Pip)"
    conda "${projectDir}/environments/dipk_graphdrp_env.yml"
    input: path local_wheels_dir
    output: path "env_ready.txt"
    script:
    """
    echo "Setting up DIPK/GraphDRP environment..."
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3
    echo "DIPK/GraphDRP Environment is ready." > env_ready.txt
    """
}

workflow run_dipk_graphdrp_workflow {
    take:
        gpu_map
        local_wheels_dir
        output_dir
        input_params

    main:
        env_ready_signal = setup_dipk_graphdrp_env(local_wheels_dir)
        
        run_DIPK_internal(gpu_map.DIPK, output_dir, input_params, env_ready_signal)

        models_ch = Channel.fromList([
            ['GATNet', 'model_GATNet_GDSC_blind_run1.model'],
            ['GAT_GCN', 'model_GAT_GCN_GDSC_blind_run1.model']
        ])
        
        run_GraphDRP_internal(gpu_map.GraphDRP, output_dir, models_ch.map{it[0]}, models_ch.map{it[1]}, input_params, env_ready_signal)
}


process run_GraphDRP_internal {
    tag "GraphDRP predict [${model_type}]"; conda "${projectDir}/environments/dipk_graphdrp_env.yml"
    input: val gpu_id; val output_dir; val model_type; val model_filename; val p; path env_ready
    script:
    def target_output = "${output_dir}/GraphDRP_predictions_${model_type}.csv"
    """
    mkdir -p "${output_dir}"
    cd ${projectDir}/GraphDRP-master
    python ./predict_all.py \\
        --drug_file "${p.drug_smiles}" \\
        --cell_file "${p.cell_file_graphdrp}" \\
        --model_path "./${model_filename}" \\
        --model_type "${model_type}" \\
        --data_dir ./mydata/ \\
        --output_file "${target_output}" \\
        --cuda_name "cuda:${gpu_id}"
    """
}

process run_DIPK_internal {
    tag "DIPK predict"; conda "${projectDir}/environments/dipk_graphdrp_env.yml"
    input: val gpu_id; val output_dir; val p; path env_ready
    script:
    def target_output = "${output_dir}/DIPK_predictions.csv"
    """
    mkdir -p "${output_dir}"
    cd ${projectDir}/DIPK-main/prog
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py \\
        --input_drugs_csv "${p.drug_smiles}" \\
        --gene_expression_file "${p.gene_exp_file}" \\
        --output_csv "${target_output}" \\
        --model_path ./result/Train.pkl \\
        --train_config_path ./TrainConfig.py \\
        --data_config_path ./DataConfig.py \\
        --molgnet_model_path ./Data/MolGNet.pt \\
        --bionic_dict_path ../Dataset/BIONIC_dict.pkl \\
        --canonical_gene_list_path ../Dataset/exp.txt \\
        --device cuda
    """
}