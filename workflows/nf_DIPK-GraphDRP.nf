#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    DIPK & GraphDRP SUB-WORKFLOW (V8.1 Final Fix)
    - The setup process now not only creates the Conda environment but also performs
      the pip installations ONCE.
    - All downstream processes now only execute the python scripts, eliminating all
      installation race conditions.
========================================================================================
*/

// ====================================================================================
//    Step 1: The setup process now handles ALL installations (Conda + Pip)
// ====================================================================================
process setup_dipk_graphdrp_env {
    tag "Setup DIPK/GraphDRP Env (Conda & Pip)"

    conda "${projectDir}/environments/dipk_graphdrp_env.yml"

    // 接收 wheel 包的路径
    input:
    path local_wheels_dir

    output:
    path "env_ready.txt"

    script:
    """
    echo "==> [SETUP] Conda environment activated. Starting pip installations..."
    
    # 【核心修复】: 在这里集中执行一次性的 pip 安装
    pip install ${local_wheels_dir}/*.whl --no-deps
    pip install torch-geometric==2.0.3
    
    echo "==> [SETUP] All pip packages installed successfully."
    echo "DIPK/GraphDRP Environment is fully prepared." > env_ready.txt
    """
}

// ====================================================================================
//    Step 2: The main workflow now passes the wheels directory to the setup process
// ====================================================================================
workflow run_dipk_graphdrp_workflow {
    take:
        gpu_id
        local_wheels_dir // 这个参数现在要传给 setup 进程
        output_dir
        input_params

    main:
        // 首先，运行环境和依赖准备进程，并把 wheel 包的路径传给它
        setup_dipk_graphdrp_env(local_wheels_dir)
        def env_ready_signal = setup_dipk_graphdrp_env.out

        run_DIPK_internal(
            gpu_id,
            output_dir,
            input_params,
            env_ready_signal // <-- 传入信号, 不再需要 local_wheels_dir
        )

        Channel
            .fromList([
                ['GATNet', 'model_GATNet_GDSC_blind_run1.model'],
                ['GAT_GCN', 'model_GAT_GCN_GDSC_blind_run1.model']
            ])
            .set{ models_ch }

        run_GraphDRP_internal(
            gpu_id,
            output_dir,
            models_ch.map { it[0] },
            models_ch.map { it[1] },
            input_params,
            env_ready_signal // <-- 传入信号, 不再需要 local_wheels_dir
        )
}

// ====================================================================================
//    Step 3: All computation processes are now simplified to only run code
// ====================================================================================
process run_GraphDRP_internal {
    tag "GraphDRP predict [${model_type}]"

    conda "${projectDir}/environments/dipk_graphdrp_env.yml"

    input:
    val gpu_id; val output_dir; val model_type; val model_filename; val p
    path env_ready

    script:
    def target_output = "${output_dir}/GraphDRP_predictions_${model_type}.csv"
    """
    # 【核心修复】: 移除了 pip install 命令
    mkdir -p ${output_dir}
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
    tag "DIPK predict"

    conda "${projectDir}/environments/dipk_graphdrp_env.yml"

    input:
    val gpu_id; val output_dir; val p
    path env_ready

    script:
    def target_output = "${output_dir}/DIPK_predictions.csv"
    """
    # 【核心修复】: 移除了 pip install 命令
    mkdir -p ${output_dir}
    
    cd ${projectDir}/DIPK-main/prog

    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    python ./predict_all.py \\
        --input_drugs_csv "${p.drug_smiles}" \\
        --gene_expression_file "${p.gene_exp_file}" \\
        --output_csv "${target_output}" \\
        --model_path ./result/Train.pkl \\
        --train_config_path ./TrainConfig.py \\
        --data_config_path ./DataConfig.py \\
        --molgnet_model_path ../Data/MolGNet.pt \\
        --bionic_dict_path ../Dataset/BIONIC_dict.pkl \\
        --canonical_gene_list_path ../Dataset/exp.txt \\
        --device cuda
    """
}