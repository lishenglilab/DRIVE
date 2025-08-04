#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    SUBWORKFLOW MODULE (Part 1 Models) - GPDRP Parameterized
========================================================================================
    - run_GPDRP process is now parameterized to run 4 different models.
    - run_part1_workflow is updated to call the parameterized GPDRP process in parallel.
----------------------------------------------------------------------------------------
*/

// ========================================================================================
//                              局部参数
// ========================================================================================
params.test_dir = "${projectDir}/test" 
params.drug_smiles              = "${params.test_dir}/drug_sample.csv"
params.drug_features_gadrp      = "${params.test_dir}/drug_with_conditions.csv"
params.drug_features_nerd       = "${params.test_dir}/drug_with_conditions.csv"
params.gene_exp                 = "${params.test_dir}/gene_sample.csv"
params.cnv                      = "${params.test_dir}/cnv_sample.csv"
params.mutation                 = "${params.test_dir}/mu_sample.csv"
params.microrna                 = "${params.test_dir}/mi_sample.csv"
params.gsva                     = "${params.test_dir}/gsva_sample.csv"

// ========================================================================================
//                                  子工作流定义
// ========================================================================================
workflow run_part1_workflow {
    take:
        gpu_id
        local_wheels_dir
        results_dir // 接收来自 main.nf 的 results_dir

    main:
        // 【【【 核心修改 1: 参数化调用 run_GPDRP 】】】
        // 1. 创建一个包含所有GPDRP模型信息的Channel
        Channel
            .fromList([
                ['GIN', 'model_GIN_GDSC_drug_blind_run2.model'],
                ['GAT', 'model_GAT_GDSC_drug_blind_run2.model'],
                ['GCN', 'model_GCN_GDSC_drug_blind_run2.model'],
                ['GINTransformer', 'model_GINTransformer_GDSC_drug_blind_run2.model']
            ])
            .set { gpdrp_models_ch }

        // 2. 将模型Channel与其它参数组合，并调用参数化的GPDRP进程
        run_GPDRP(
            gpu_id, 
            local_wheels_dir, 
            results_dir,
            gpdrp_models_ch.map { it[0] }, // 传递模型类型
            gpdrp_models_ch.map { it[1] }  // 传递模型文件名
        )
        
        // 其他模型的调用保持不变
        run_BANDRP(gpu_id, local_wheels_dir, results_dir)
        run_DeepTTC(gpu_id, local_wheels_dir, results_dir)
        run_GADRP(gpu_id, local_wheels_dir, results_dir)
        run_NeRD(gpu_id, local_wheels_dir, results_dir)
        run_paccmann(gpu_id, local_wheels_dir, results_dir)
        run_Precily(gpu_id, local_wheels_dir, results_dir)
}

// ========================================================================================
//                                  流程定义
// ========================================================================================

process run_BANDRP {
    tag "BANDRP_predict"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    
    script:
    def target_output = "${results_dir}/bandrp_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/BANDRP-main
    python ./predict_all.py \\
        --model_path ./github_upload/output_dir/db1/model.pt \\
        --new_drugs_csv "${params.drug_smiles}" \\
        --exp_path "${params.gene_exp}" \\
        --mut_path "${params.mutation}" \\
        --cnv_path "${params.cnv}" \\
        --output_csv "${target_output}" \\
        --drug_batch_size 100000 \\
        --cuda_id ${gpu_id}
    """
}

process run_DeepTTC {
    tag "DeepTTC_predict"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    
    script:
    def target_output = "${results_dir}/DeepTTC_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/DeepTTC
    python ./predict_all.py \\
        --new_drug_file ${params.drug_smiles} \\
        --new_cell_line_file ${params.gene_exp} \\
        --training_gene_list_file "./mydata/expt.txt" \\
        --model_dir "./DeepTTC" \\
        --vocab_dir "./ESPF" \\
        --output_file "${target_output}"
    """
}

process run_GADRP {
    tag "GADRP_predict"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    
    script:
    def target_output = "${results_dir}/GADRP_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/GADRP-main
    python ./predict_all.py \\
        --new_drug_feature_file ${params.drug_features_gadrp} \\
        --new_exp_file ${params.gene_exp} \\
        --new_cn_file ${params.cnv} \\
        --new_meth_file ${params.mutation} \\
        --new_mirna_file ${params.microrna} \\
        --model_path ./model/saved_models/best_model_drug_blind_fold2.pth \\
        --output_file "${target_output}"
    """
}

// 【【【 核心修改 2: 参数化 run_GPDRP 进程 】】】
process run_GPDRP {
    // 标签现在会动态显示正在运行的模型
    tag "GPDRP predict [${model_type}]"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    // 新增模型相关的输入
    val model_type
    val model_filename
    
    script:
    // 动态生成带后缀的输出文件名
    def target_output = "${results_dir}/GPDRP_predictions_${model_type}.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/GPDRP
    python ./predict_all.py \\
        --smiles_file "${params.drug_smiles}" \\
        --new_cell_line_file "${params.gsva}" \\
        --training_gene_expression_file ./mydata/exp.txt \\
        --model_file "./output/models/${model_filename}" \\
        --output_file "${target_output}" \\
        --model_type "${model_type}" \\
        --cuda_name "cuda:${gpu_id}"
    """
}

process run_NeRD {
    tag "NeRD_predict"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    
    script:
    def target_output = "${results_dir}/NeRD_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/NeRD-main
    python ./predict_all.py \\
        --model_path ./result3/model.model \\
        --output_file "${target_output}" \\
        --drugs_input_file ${params.drug_smiles} \\
        --new_mirna_file ${params.microrna} \\
        --new_cnv_file ${params.cnv} \\
        --precomputed_fingerprint_file ${params.drug_features_nerd} \\
        --train_mirna_file ./mydata/cell_line/mirna.csv \\
        --train_cnv_raw_file ./mydata/cell_line/cnv_489.csv \\
        --ic50_scaling_params_file ./mydata/ic5o_scaling_parameters.json
    """
}

process run_paccmann {
    tag "paccmann_predict"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    
    script:
    def target_output = "${results_dir}/paccmann_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/paccmann_predictor-master
    python ./predict_all.py \\
        --predict_smiles_filepath "${params.drug_smiles}" \\
        --gep_filepath "${params.gene_exp}" \\
        --model_run_path ./paccman_training_runs/paccmann_train_1747977832 \\
        --gene_filepath_spec ./data/2128_genes.pkl \\
        --smiles_language_filepath ./paccmann/smiles_language.pkl \\
        --output_filepath "${target_output}"
    """
}

process run_Precily {
    tag "Precily_predict"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    
    script:
    def target_output = "${results_dir}/Precily_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3

    cd ${projectDir}/Precily-v1.0.0/Pathway_based
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py \\
        --input_drugs_file "${params.drug_smiles}" \\
        --input_cell_lines_file "${params.gsva}" \\
        --models_dir . \\
        --output_file "${target_output}" \\
        --drug_embedding_file ../mydata/utils/drug.pubchem.canon.l8.ws20.txt \\
        --elements_file ../mydata/utils/elements.txt \\
        --cell_features_template_file ../mydata/mycell_gsva2.csv
    """
}