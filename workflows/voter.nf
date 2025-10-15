#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    Ensemble Analysis Sub-workflow (V6)
    - Replaces the old voter with the new, more powerful run_ensemble.py script.
========================================================================================
*/

workflow run_ensemble {
    take:
        prediction_files // A channel containing all generated prediction CSVs
        output_dir       // The main output directory
        model_pkl_file   // Path to the .pkl file for Mode 0
        weight_csv_file  // Path to the weight.csv file for Mode 1

    main:
        run_ensemble_analysis(
            prediction_files,
            output_dir,
            model_pkl_file,
            weight_csv_file
        )
}

process run_ensemble_analysis {
    tag "Ensemble Analysis"

    input:
    path prediction_files // Ensures this process runs after predictions are made
    val output_dir
    path model_pkl
    path weight_csv

    script:
    """
    echo "--- All prediction tasks completed. Starting ensemble analysis. ---"

    # The run_ensemble.py script expects all input CSVs, pkl, and weight files
    # to be in its current working directory. Nextflow stages them here automatically.
    
    echo "--- [1/2] Running Mode 0: ML Model Prediction ---"
    python ${projectDir}/scripts/run_ensemble.py --mode 0 --model_pkl ${model_pkl}

    echo "--- [2/2] Running Mode 1: Weighted Average Reports ---"
    python ${projectDir}/scripts/run_ensemble.py --mode 1 --weight_file ${weight_csv} --top_n 30

    echo "--- Ensemble analysis finished. Moving results to main output directory. ---"
    
    # Move the generated output directory into the main results folder
    mv ./ensemble_outputs/* ${output_dir}/
    """
}