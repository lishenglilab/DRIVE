#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    主路由工作流 (已集成Voter)
========================================================================================
*/

// ========================================================================================
//                             顶层全局参数
// ========================================================================================
params.entry = ''
params.gpu_id = 0
params.test_dir = "${projectDir}/test"
params.results_dir = "${projectDir}/results"
params.local_wheels_cp37 = "${projectDir}/local_wheels/cp37"
params.local_wheels_cp38 = "${projectDir}/local_wheels/cp38"

// --- Voter 相关参数 ---
params.predict_lnic50 = 0
params.BANDRP = 1
params.DeepAEG = 1
params.DeepCDR = 1
params.DeepTTA = 1
params.DIPK = 1
params.GADRP = 1
params.GPDRP_GAT = 1
params.GPDRP_GCN = 1
params.GPDRP_GIN = 1
params.GPDRP_GINTransformer = 1
params.GraphDRP_GATNet = 1
params.GraphDRP_GAT_GCN = 1
params.GraphDRP_GCNNet = 1
params.GraphDRP_GINConvNet = 1
params.NERD = 1
params.paccmann = 1
params.Precily = 1

// ========================================================================================
//                     包含子工作流和流程
// ========================================================================================
include { run_part1_workflow } from './workflows/nf_part1.nf'
include { run_DeepAEG as run_deepaeg_process } from './workflows/nf_DeepAEG.nf'
include { run_deepcdr_workflow } from './workflows/nf_deepcdr.nf'
include { run_dipk_graphdrp_workflow } from './workflows/nf_DIPK-GraphDRP.nf'
include { run_voter } from './workflows/voter.nf'

// ========================================================================================
//                                主工作流 (MAIN WORKFLOW)
// ========================================================================================
workflow {
    log.info """
    ===================================================
    DRP Main Workflow Started
    ===================================================
    Entry Point: ${params.entry}
    Test Data Directory: ${params.test_dir}
    Results Directory: ${params.results_dir}
    ===================================================
    """.stripIndent()

    file(params.results_dir).mkdirs()

    // 收集所有Voter参数到一个Map中
    def voter_params = [
        predict_lnic50: params.predict_lnic50,
        BANDRP: params.BANDRP,
        DeepAEG: params.DeepAEG,
        DeepCDR: params.DeepCDR,
        DeepTTA: params.DeepTTA,
        DIPK: params.DIPK,
        GADRP: params.GADRP,
        GPDRP_GAT: params.GPDRP_GAT,
        GPDRP_GCN: params.GPDRP_GCN,
        GPDRP_GIN: params.GPDRP_GIN,
        GPDRP_GINTransformer: params.GPDRP_GINTransformer,
        GraphDRP_GATNet: params.GraphDRP_GATNet,
        GraphDRP_GAT_GCN: params.GraphDRP_GAT_GCN,
        GraphDRP_GCNNet: params.GraphDRP_GCNNet,
        GraphDRP_GINConvNet: params.GraphDRP_GINConvNet,
        NERD: params.NERD,
        paccmann: params.paccmann,
        Precily: params.Precily
    ]

    // --- 使用 'if' 语句进行流程路由 ---
    if (params.entry == 'voter') {
        // 【新增】单独运行voter的入口
        log.info "--> Voter only workflow selected."
        run_voter(Channel.fromPath("${params.results_dir}/*_predictions*.csv").collect(), params.results_dir, voter_params)

    } else if (params.entry == 'deepaeg') {
        run_deepaeg_process(params.gpu_id, params.results_dir)

    } else if (params.entry == 'deepcdr') {
        run_deepcdr_workflow(params.results_dir)

    } else if (params.entry == 'dipk_graphdrp') {
        run_dipk_graphdrp_workflow(params.gpu_id, file(params.local_wheels_cp38), params.results_dir)

    } else if (params.entry == 'part1') {
        run_part1_workflow(params.gpu_id, file(params.local_wheels_cp37), params.results_dir)

    } else {
        log.info """
        
        ERROR: No valid workflow entry point specified.
        Please provide an entry point using the --entry parameter or a profile.
        
        Available entry points are: 'voter', 'deepaeg', 'deepcdr', 'dipk_graphdrp', 'part1'
        
        """.stripIndent()
        exit 1
    }
}

// ========================================================================================
//                            工作流完成时的回调函数
// ========================================================================================
workflow.onComplete {
    if (workflow.success) {
        log.info """
        
        ===================================================
        Workflow Completed Successfully!
        ===================================================
        Check the results in the '${params.results_dir}' directory.
        
        """.stripIndent()
    } else {
        log.error """
        
        ===================================================
        Workflow FAILED!
        ===================================================
        Exit Status : ${workflow.exitStatus}
        Error Report: ${workflow.errorReport}
        Work Dir    : ${workflow.workDir}
        ===================================================
        Please check the logs in the work directory for details.
        
        """.stripIndent()
    }
}