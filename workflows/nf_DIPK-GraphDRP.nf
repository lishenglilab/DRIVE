#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    DIPK & GraphDRP SUB-WORKFLOW (with Parallel GraphDRP)
========================================================================================
    - Defines a self-contained sub-workflow `run_dipk_graphdrp_workflow`.
    - Internally defines `run_GraphDRP_internal` and `run_DIPK_internal` processes.
    - `GraphDRP` is executed in PARALLEL for all 4 models using a Channel.
----------------------------------------------------------------------------------------
*/

// ========================================================================================
//                                  内部进程定义
// ========================================================================================

// 内部GraphDRP进程，现在接收模型信息作为输入
process run_GraphDRP_internal {
    tag "GraphDRP predict [${model_type}]"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir
    val model_type
    val model_filename

    script:
    def target_output = "${results_dir}/GraphDRP_predictions_${model_type}.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3
    echo "--- Dependencies installed. Starting GraphDRP prediction for model: ${model_type} ---"
    cd ${projectDir}/GraphDRP-master
    python ./predict_all.py \\
        --drug_file "${projectDir}/test/drug_sample.csv" \\
        --cell_file "${projectDir}/test/mu_sample.csv" \\
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

// 内部DIPK进程
process run_DIPK_internal {
    tag "DIPK predict"

    input:
    val gpu_id
    path local_wheels_dir
    val results_dir

    script:
    def bionic_dict_path = "${projectDir}/DIPK-main/Dataset/BIONIC_dict.pkl"
    def gene_list_path = "${projectDir}/DIPK-main/Dataset/exp.txt"
    def target_output = "${results_dir}/DIPK_predictions.csv"
    """
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3
    cd ${projectDir}/DIPK-main/fold=0_model=0
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python ./predict_all.py \\
        --input_drugs_csv "${projectDir}/test/drug_sample.csv" \\
        --gene_expression_file "${projectDir}/test/gene_sample.csv" \\
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

// ========================================================================================
//                                  可被外部引用的工作流
// ========================================================================================

// 【【【 核心：定义一个总的子工作流 】】】
workflow run_dipk_graphdrp_workflow {
    take:
        gpu_id
        local_wheels_dir
        results_dir
        
    main:
        // --- DIPK部分 ---
        // 调用DIPK进程，它会独立运行
        run_DIPK_internal(gpu_id, local_wheels_dir, results_dir)

        // --- GraphDRP并行处理部分 ---
        // 1. 创建包含所有模型信息的Channel
        Channel
            .fromList([
                ['GCNNet', 'model_GCNNet_GDSC_blind_run1.model'],
                ['GINConvNet', 'model_GINConvNet_GDSC_blind_run1.model'],
                ['GATNet', 'model_GATNet_GDSC_blind_run1.model'],
                ['GAT_GCN', 'model_GAT_GCN_GDSC_blind_run1.model']
            ])
            .set{ models_ch }

        // 2. 将模型Channel与其他参数组合，并调用内部的GraphDRP进程
        //    Nextflow会为Channel中的每一个元素启动一个并行的任务
        run_GraphDRP_internal(
            gpu_id,
            local_wheels_dir,
            results_dir,
            models_ch.map { it[0] }, // 传递模型类型
            models_ch.map { it[1] }  // 传递模型文件名
        )
}