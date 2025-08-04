#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    VOTER SUB-WORKFLOW
========================================================================================
    - Defines a process to run the ensemble voter script.
    - Dynamically builds the command line arguments based on workflow params.
----------------------------------------------------------------------------------------
*/

process run_voter {
    tag "Ensemble Voting"

    // 这个进程的输入是上游所有任务的完成信号
    input:
    val done_signal
    val results_dir
    val voter_params // 接收一个包含所有voter参数的map

    script:
    // 动态构建命令行参数字符串
    def voter_args = voter_params.collect { key, value -> "--${key} ${value}" }.join(' ')
    """
    echo "--- All prediction tasks completed. Starting the voter process. ---"
    
    # 切换到结果目录，因为voter.py需要在这里找到所有输入文件
    cd ${results_dir}

    echo "Running voter with the following arguments:"
    echo "python voter.py ${voter_args}"

    python voter.py ${voter_args}

    echo "--- Voter process finished. ---"
    """
}
