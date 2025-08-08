#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    VOTER SUB-WORKFLOW (V5)
========================================================================================
*/

process run_voter {
    tag "Ensemble Voting"

    input:
    val done_signal
    val output_dir
    val voter_params

    script:
    def voter_args = voter_params.collect { key, value -> "--${key} ${value}" }.join(' ')
    """
    echo "--- All prediction tasks completed. Starting the voter process. ---"
    
    # 切换到结果目录
    cd ${output_dir}

    echo "Running voter with the following arguments:"
    # 假设voter.py位于项目的某个固定位置，例如 {projectDir}/scripts/
    # 如果voter.py就在工作目录，可以直接运行 python voter.py
    echo "python ${projectDir}/scripts/voter.py ${voter_args}"

    python ${projectDir}/scripts/voter.py ${voter_args}

    echo "--- Voter process finished. ---"
    """
}