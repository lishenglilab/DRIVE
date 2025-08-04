#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    DeepCDR 子工作流 (修正输出路径)
========================================================================================
*/

// 定义子工作流的入口
workflow run_deepcdr_workflow {
    take:
        results_dir // 从 main.nf 接收全局结果目录

    main:
        // 直接调用流程
        run_DeepCDR(results_dir)
}

// 定义执行预测的流程
process run_DeepCDR {
    tag "DeepCDR_Prediction"

    // --- 关键修正 ---
    // 将输出目录直接设置为全局的 results_dir，不再创建 deepcdr 子目录
    publishDir "${params.results_dir}", mode: 'copy', pattern: "DeepCDR_predictions.csv"

    // 输入参数，虽然在这里不直接使用，但保留以保持接口清晰
    input:
    val results_dir 

    // 声明此流程将产出的文件名
    output:
    path "DeepCDR_predictions.csv" 

    script:
    // --- 定义所有输入文件路径 ---
    def model_file = "${projectDir}/DeepCDR/prog/saved_models/bd1.h5"
    def drugs_file = "${params.test_dir}/drug_sample.csv"
    def mut_file = "${params.test_dir}/mu_sample.csv"
    def gexp_file = "${params.test_dir}/gene_sample.csv"
    def methy_file = "${params.test_dir}/mu_sample.csv" // 假设使用突变数据作为代理
    def align_gexp_file = "${projectDir}/DeepCDR/depmap/CCLE/exp.csv"
    def align_mut_file = "${projectDir}/DeepCDR/depmap/CCLE/mu.csv"

    // 定义输出文件名（与 output 块中声明的保持一致）
    def output_file = "DeepCDR_predictions.csv"
    """
    echo "--- Starting DeepCDR Prediction ---"
    
    python ${projectDir}/DeepCDR/prog/predict_all.py \\
        --model_file "${model_file}" \\
        --drugs_file "${drugs_file}" \\
        --mut_file "${mut_file}" \\
        --gexp_file "${gexp_file}" \\
        --methy_file "${methy_file}" \\
        --align_gexp_file "${align_gexp_file}" \\
        --align_mut_file "${align_mut_file}" \\
        --output_file "${output_file}"

    if [ ! -f "${output_file}" ]; then
        echo "ERROR: DeepCDR script finished, but the output file was NOT created." >&2
        exit 1
    else
        echo "--- DeepCDR Prediction finished successfully. Output file: ${output_file} ---"
    fi
    """
}