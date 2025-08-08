#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    SUBWORKFLOW MODULE (Part 1 Models) - V5 Parameterized
========================================================================================
*/

workflow run_part1_workflow {
    take:
        gpu_id
        local_wheels_dir
        output_dir

    main:
        Channel
            .fromList([
                ['GIN', 'model_GIN_GDSC_drug_blind_run2.model'],
                ['GAT', 'model_GAT_GDSC_drug_blind_run2.model'],
                ['GCN', 'model_GCN_GDSC_drug_blind_run2.model'],
                ['GINTransformer', 'model_GINTransformer_GDSC_drug_blind_run2.model']
            ])
            .set { gpdrp_models_ch }

        run_GPDRP(
            gpu_id, 
            local_wheels_dir, 
            output_dir,
            gpdrp_models_ch.map { it[0] },
            gpdrp_models_ch.map { it[1] }
        )
        
        run_BANDRP(gpu_id, local_wheels_dir, output_dir)
        run_DeepTTC(gpu_id, local_wheels_dir, output_dir)
        run_GADRP(gpu_id, local_wheels_dir, output_dir)
        run_NeRD(gpu_id, local_wheels_dir, output_dir)
        run_paccmann(gpu_id, local_wheels_dir, output_dir)
        run_Precily(gpu_id, local_wheels_dir, output_dir)
}

process run_BANDRP {
    tag "BANDRP_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir
    script:
    def target_output = "${output_dir}/bandrp_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/BANDRP-main
    python ./predict_all.py --model_path ./github_upload/output_dir/db1/model.pt --new_drugs_csv "${params.drug_smiles}" --exp_path "${params.gene_exp_file}" --mut_path "${params.mutation_file}" --cnv_path "${params.cnv_file}" --output_csv "${target_output}" --drug_batch_size 100000 --cuda_id ${gpu_id}
    """
}

process run_DeepTTC {
    tag "DeepTTC_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir
    script:
    def target_output = "${output_dir}/DeepTTC_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/DeepTTC
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --new_drug_file ${params.drug_smiles} --new_cell_line_file ${params.gene_exp_file} --training_gene_list_file "./mydata/expt.txt" --model_dir "./DeepTTC" --vocab_dir "./ESPF" --output_file "${target_output}"
    """
}

process run_GADRP {
    tag "GADRP_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir
    script:
    def target_output = "${output_dir}/GADRP_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/GADRP-main
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --new_drug_feature_file ${params.drug_features_file} --new_exp_file ${params.gene_exp_file} --new_cn_file ${params.cnv_file} --new_meth_file ${params.mutation_file} --new_mirna_file ${params.microrna_file} --model_path ./model/saved_models/best_model_drug_blind_fold2.pth --output_file "${target_output}"
    """
}

process run_GPDRP {
    tag "GPDRP predict [${model_type}]"
    input:
    val gpu_id; path local_wheels_dir; val output_dir; val model_type; val model_filename
    script:
    def target_output = "${output_dir}/GPDRP_predictions_${model_type}.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/GPDRP
    python ./predict_all.py --smiles_file "${params.drug_smiles}" --new_cell_line_file "${params.gsva_file}" --training_gene_expression_file ./mydata/exp.txt --model_file "./output/models/${model_filename}" --output_file "${target_output}" --model_type "${model_type}" --cuda_name "cuda:${gpu_id}"
    """
}

process run_NeRD {
    tag "NeRD_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir
    script:
    def target_output = "${output_dir}/NeRD_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/NeRD-main
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --model_path ./result3/model.model --output_file "${target_output}" --drugs_input_file ${params.drug_smiles} --new_mirna_file ${params.microrna_file} --new_cnv_file ${params.cnv_file} --precomputed_fingerprint_file ${params.drug_features_file} --train_mirna_file ./mydata/cell_line/mirna.csv --train_cnv_raw_file ./mydata/cell_line/cnv_489.csv --ic50_scaling_params_file ./mydata/ic5o_scaling_parameters.json
    """
}

process run_paccmann {
    tag "paccmann_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir
    script:
    def target_output = "${output_dir}/paccmann_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/paccmann_predictor-master
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --predict_smiles_filepath "${params.drug_smiles}" --gep_filepath "${params.gene_exp_file}" --model_run_path ./paccman_training_runs/paccmann_train_1747977832 --gene_filepath_spec ./data/2128_genes.pkl --smiles_language_filepath ./paccmann/smiles_language.pkl --output_filepath "${target_output}"
    """
}

process run_Precily {
    tag "Precily_predict"
    input:
    val gpu_id; path local_wheels_dir; val output_dir
    script:
    def target_output = "${output_dir}/Precily_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps; pip install torch-geometric==2.0.3
    cd ${projectDir}/Precily-v1.0.0/Pathway_based
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py --input_drugs_file "${params.drug_smiles}" --input_cell_lines_file "${params.gsva_file}" --models_dir . --output_file "${target_output}" --drug_embedding_file ../mydata/utils/drug.pubchem.canon.l8.ws20.txt --elements_file ../mydata/utils/elements.txt --cell_features_template_file ../mydata/mycell_gsva2.csv
    """
}