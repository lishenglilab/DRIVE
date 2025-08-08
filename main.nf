#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    主路由工作流 (V5 - 最终版)
========================================================================================
    - 所有用户输入文件参数在此处定义了默认路径（指向test数据）。
    - 任何参数都可以通过命令行（如 --drug_smiles /path/to/file）被覆盖。
    - 为主要模型组分配独立的GPU ID。
========================================================================================
*/

// ========================================================================================
//                             顶层全局参数 (带默认值)
// ========================================================================================

// --- 工作流控制与环境 ---
params.entry = ''
params.output_dir = "${projectDir}/results" // 统一的、可自定义的输出目录

// --- GPU 分配控制 ---
params.gpu_part1         = 0 // 分配给 part1 组模型的 GPU ID
params.gpu_deepaeg       = 0 // 分配给 DeepAEG 的 GPU ID
params.gpu_deepcdr       = 0 // 分配给 DeepCDR 的 GPU ID
params.gpu_dipk_graphdrp = 0 // 分配给 DIPK & GraphDRP 组的 GPU ID

// --- 用户输入文件路径 (此处提供默认值，方便快速测试) ---
params.drug_smiles              = "${projectDir}/test/drug_sample.csv"
params.all_in_one_deepaeg       = "${projectDir}/test/al_sample.csv"
params.mutation_file            = "${projectDir}/test/mu_sample.csv"
params.gene_exp_file            = "${projectDir}/test/gene_sample.csv"
params.cnv_file                 = "${projectDir}/test/cnv_sample.csv"
params.microrna_file            = "${projectDir}/test/mi_sample.csv"
params.gsva_file                = "${projectDir}/test/gsva_sample.csv"
params.drug_features_file       = "${projectDir}/test/drug_with_conditions.csv" // GADRP 和 NeRD 共享
params.cell_file_graphdrp       = "${projectDir}/test/mu_sample.csv"

// --- 环境与依赖 ---
params.local_wheels_cp37 = "${projectDir}/local_wheels/cp37"
params.local_wheels_cp38 = "${projectDir}/local_wheels/cp38"

// --- Voter 相关参数 ---
params.predict_lnic50 = 0
params.BANDRP = 1; params.DeepAEG = 1; params.DeepCDR = 1; params.DeepTTA = 1
params.DIPK = 1; params.GADRP = 1; params.GPDRP_GAT = 1; params.GPDRP_GCN = 1
params.GPDRP_GIN = 1; params.GPDRP_GINTransformer = 1; params.GraphDRP_GATNet = 1
params.GraphDRP_GAT_GCN = 1; params.GraphDRP_GCNNet = 1; params.GraphDRP_GINConvNet = 1
params.NERD = 1; params.paccmann = 1; params.Precily = 1

log.info """
===================================================
DRP Main Workflow Started
===================================================
Entry Point          : ${params.entry}
Output Directory     : ${params.output_dir}
--- GPU Allocation ---
Part1 Models         : GPU ${params.gpu_part1}
DeepAEG              : GPU ${params.gpu_deepaeg}
DeepCDR              : GPU ${params.gpu_deepcdr}
DIPK & GraphDRP      : GPU ${params.gpu_dipk_graphdrp}
--- Core Input Files ---
Drug SMILES          : ${params.drug_smiles}
Gene Expression      : ${params.gene_exp_file}
Mutation             : ${params.mutation_file}
===================================================
""".stripIndent()

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
    file(params.output_dir).mkdirs()

    def voter_params = [
        predict_lnic50: params.predict_lnic50, BANDRP: params.BANDRP, DeepAEG: params.DeepAEG,
        DeepCDR: params.DeepCDR, DeepTTA: params.DeepTTA, DIPK: params.DIPK, GADRP: params.GADRP,
        GPDRP_GAT: params.GPDRP_GAT, GPDRP_GCN: params.GPDRP_GCN, GPDRP_GIN: params.GPDRP_GIN,
        GPDRP_GINTransformer: params.GPDRP_GINTransformer, GraphDRP_GATNet: params.GraphDRP_GATNet,
        GraphDRP_GAT_GCN: params.GraphDRP_GAT_GCN, GraphDRP_GCNNet: params.GraphDRP_GCNNet,
        GraphDRP_GINConvNet: params.GraphDRP_GINConvNet, NERD: params.NERD, paccmann: params.paccmann,
        Precily: params.Precily
    ]

    if (params.entry == 'voter') {
        run_voter(Channel.fromPath("${params.output_dir}/*_predictions*.csv").collect(), params.output_dir, voter_params)
    } else if (params.entry == 'deepaeg') {
        run_deepaeg_process(params.gpu_deepaeg, params.output_dir)
    } else if (params.entry == 'deepcdr') {
        run_deepcdr_workflow(params.gpu_deepcdr, params.output_dir)
    } else if (params.entry == 'dipk_graphdrp') {
        run_dipk_graphdrp_workflow(params.gpu_dipk_graphdrp, file(params.local_wheels_cp38), params.output_dir)
    } else if (params.entry == 'part1') {
        run_part1_workflow(params.gpu_part1, file(params.local_wheels_cp37), params.output_dir)
    } else {
        log.error """
        ERROR: No valid workflow entry point specified. Please use --entry.
        Available: 'voter', 'deepaeg', 'deepcdr', 'dipk_graphdrp', 'part1'
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
        Results are in '${params.output_dir}'.
        ===================================================
        """.stripIndent()
    } else {
        log.error """
        ===================================================
        Workflow FAILED!
        ===================================================
        Exit Status : ${workflow.exitStatus}
        Error Report: ${workflow.errorReport}
        Work Dir    : ${workflow.workDir}
        Please check the logs in the work directory for details.
        ===================================================
        """.stripIndent()
    }
}