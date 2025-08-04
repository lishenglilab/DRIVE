#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    MODULAR WORKFLOW (DeepAEG Only) - WITH ERROR REPORTING
    此版本会捕获并报告 Python 脚本的执行失败，而不是静默成功。
========================================================================================
*/

// ========================================================================================
//                              P A R A M E T E R S
// ========================================================================================
params.gpu_id = 0
params.drug_smiles = "${params.test_dir}/drug_sample.csv"
params.all_in_one = "${params.test_dir}/al_sample.csv"
// 结果目录现在可以从 main.nf 传入，这里提供一个默认值
params.output_dir = "${projectDir}/results" 

log.info """
    DeepAEG PREDICTION WORKFLOW (with Error Reporting)
    =================================
    Project Base Directory: ${projectDir}
    Test Data Directory   : ${params.test_dir}
    Target GPU ID         : ${params.gpu_id}
    Output Directory      : ${params.output_dir}
    =================================
    """
    .stripIndent()

// ========================================================================================
//                                  W O R K F L O W
// ========================================================================================
// 这个 workflow 块只在独立运行此文件时有效。
// 当被 main.nf include 时，它不会被执行。
workflow {
    run_DeepAEG(params.gpu_id, params.output_dir)
}

// ========================================================================================
//                                  P R O C E S S
// ========================================================================================
process run_DeepAEG {
    tag "DeepAEG_predict_with_check"
    
    input:
    val gpu_id
    val output_dir // 从 workflow 接收结果目录

    script:
    // 将输出文件名定义为变量，方便引用
    def output_filename = "DeepAEG_predictions.csv"
    def output_filepath = "${output_dir}/${output_filename}"
    """
    # 1. 提前创建好目标目录，以防万一
    mkdir -p ${output_dir}

    # 2. 切换到脚本所在的目录 (这是原始逻辑)
    cd ${projectDir}/DeepAEG-main/prog/
    
    # 3. 运行 python 脚本，并将 -output_file 参数直接指定为最终的绝对路径
    python ./predict_all.py \\
        -model_path ./MyBestDeepAEG_0.7789226722858869.h5 \\
        -new_drug_file ${params.drug_smiles} \\
        -gene_info_file ${params.all_in_one} \\
        -output_file "${output_filepath}" \\
        -vocab_path_bpe ./ESPF/drug_codes_chembl_freq_1500.txt \\
        -subword_csv_path_bpe ./ESPF/subword_units_map_chembl_freq_1500.csv \\
        -gpu_id ${gpu_id}

    # 4.【关键的错误检查步骤】
    # 检查 python 脚本是否真的创建了输出文件
    if [ ! -f "${output_filepath}" ]; then
        # 如果文件不存在，打印一条明确的错误信息到标准错误流
        echo "ERROR: Python script finished, but the output file was NOT created at '${output_filepath}'." >&2
        echo "This indicates a silent failure within the Python script. Please check the logs above for Python errors." >&2
        # 用非零退出码退出，强制让 Nextflow 报告失败
        exit 1
    fi
    """
}