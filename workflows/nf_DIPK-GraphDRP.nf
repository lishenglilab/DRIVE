#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    DIPK & GraphDRP SUB-WORKFLOW (V5 - Parameterized)
========================================================================================
*/

process run_GraphDRP_internal {
    tag "GraphDRP predict [${model_type}]"

    input:
    val gpu_id
    path local_wheels_dir
    val output_dir
    val model_type
    val model_filename

    script:
    def target_output = "${output_dir}/GraphDRP_predictions_${model_type}.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/GraphDRP-master
    python ./predict_all.py \\
        --drug_file "${params.drug_smiles}" \\
        --cell_file "${params.cell_file_graphdrp}" \\
        --model_path "./${model_filename}" \\
        --model_type "${model_type}" \\
        --data_dir ./mydata/ \\
        --output_file "${target_output}" \\
        --cuda_name "cuda:${gpu_id}"

    if [ ! -f "${target_output}" ]; then
        echo "ERROR: GraphDRP script for model '${model_type}' failed." >&2
        exit 1
    fi
    """
}

process run_DIPK_internal {
    tag "DIPK predict"

    input:
    val gpu_id
    path local_wheels_dir
    val output_dir

    script:
    def bionic_dict_path = "${projectDir}/DIPK-main/Dataset/BIONIC_dict.pkl"
    def gene_list_path = "${projectDir}/DIPK-main/Dataset/exp.txt"
    def target_output = "${output_dir}/DIPK_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/DIPK-main/fold=0_model=0
    export CUDA_VISIBLE_DEVICES=${gpu_id}

    python ./predict_all.py \\
        --input_drugs_csv "${params.drug_smiles}" \\
        --gene_expression_file "${params.gene_exp_file}" \\
        --output_csv "${target_output}" \\
        --model_path ./result/Train.pkl \\
        --train_config_path ./TrainConfig.py \\
        --data_config_path ./DataConfig.py \\
        --molgnet_model_path ./Data/MolGNet.pt \\
        --bionic_dict_path "${bionic_dict_path}" \\
        --canonical_gene_list_path "${gene_list_path}" \\
        --device cuda

    if [ ! -f "${target_output}" ]; then
        echo "ERROR: DIPK script failed." >&2
        exit 1
    fi
    """
}

workflow run_dipk_graphdrp_workflow {
    take:
        gpu_id
        local_wheels_dir
        output_dir
        
    main:
        run_DIPK_internal(gpu_id, local_wheels_dir, output_dir)

        Channel
            .fromList([
                ['GCNNet', 'model_GCNNet_GDSC_blind_run1.model'],
                ['GINConvNet', 'model_GINConvNet_GDSC_blind_run1.model'],
                ['GATNet', 'model_GATNet_GDSC_blind_run1.model'],
                ['GAT_GCN', 'model_GAT_GCN_GDSC_blind_run1.model']
            ])
            .set{ models_ch }

        run_GraphDRP_internal(
            gpu_id,
            local_wheels_dir,
            output_dir,
            models_ch.map { it[0] }, // model_type
            models_ch.map { it[1] }  // model_filename
        )
}