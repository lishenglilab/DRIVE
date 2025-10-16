#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    Ensemble Analysis Sub-workflow (V7.0 - Aligned with updated voter.py)
    - Removed dependency on weight_csv_file.
    - Split analysis into two separate processes for Mode 0 and Mode 1 for clarity.
    - Both processes use the RandomForest model predictions as their basis.
========================================================================================
*/

workflow run_ensemble {
    take:
        prediction_files // Channel of all prediction CSVs
        output_dir
        model_pkl_file   // Path to the .pkl file

    main:
        def voter_script = file("${projectDir}/voter.py")
        
        // Run Mode 0: Generate the single large prediction file
        generate_global_predictions(
            prediction_files,
            output_dir,
            model_pkl_file,
            voter_script
        )

        // Run Mode 1: Generate Top-K reports for each cell line
        generate_top_k_reports(
            prediction_files,
            output_dir,
            model_pkl_file,
            voter_script
        )
}

process generate_global_predictions {
    tag "Ensemble Mode 0: Global Predictions"

    conda "${projectDir}/environments/part1_env.yml"

    input:
    path prediction_files
    val output_dir
    path model_pkl
    path voter_script

    script:
    """
    echo "--- [VOTER] Activating cached environment from Part 1 run. ---"
    echo "--- Running Mode 0: Generating global ML-based predictions. ---"

    python ${voter_script} --mode 0 --model_pkl ${model_pkl}

    echo "--- Mode 0 finished. Moving results to main output directory. ---"
    mv ./ensemble_outputs/ml_ensemble_predictions.csv ${output_dir}/
    """
}

process generate_top_k_reports {
    tag "Ensemble Mode 1: Top-K Reports"

    conda "${projectDir}/environments/part1_env.yml"

    input:
    path prediction_files
    val output_dir
    path model_pkl
    path voter_script

    // 使用 params 来控制 top_k 的值，增加灵活性
    // 可以在 nextflow.config 或命令行 (--top_k 10) 中修改
    params.top_k = 5

    script:
    """
    echo "--- [VOTER] Activating cached environment from Part 1 run. ---"
    echo "--- Running Mode 1: Generating Top-${params.top_k} reports per cell line. ---"

    python ${voter_script} --mode 1 --model_pkl ${model_pkl} --top_k ${params.top_k}

    echo "--- Mode 1 finished. Moving results to main output directory. ---"
    # 移动整个 reports 文件夹
    mv ./ensemble_outputs/cell_line_reports ${output_dir}/
    """
}