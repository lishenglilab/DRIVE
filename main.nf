#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    Main Routing Workflow (V7.3 - Aligned with updated voter.py)
    - Implements explicit dependency injection for all sub-workflows.
    - Removed weight_file parameter as it's no longer used by the ensemble script.
========================================================================================
*/

// ========================================================================================
//                             Top-Level Global Parameters
// ========================================================================================

// --- Workflow Control & Environment ---
params.entry = ''
params.output_dir = "${projectDir}/results"

// --- GPU Allocation Control ---
params.gpu_part1         = 0 // GPU ID for the part1 model group
params.gpu_dipk_graphdrp = 0 // GPU ID for the DIPK & GraphDRP group

// --- User Input File Paths (with defaults for testing) ---
params.drug_smiles              = "${projectDir}/test/drug_sample.csv"
params.mutation_file            = "${projectDir}/test/mu_sample.csv"
params.gene_exp_file            = "${projectDir}/test/gene_sample.csv"
params.cnv_file                 = "${projectDir}/test/cnv_sample.csv"
params.microrna_file            = "${projectDir}/test/mi_sample.csv"
params.gsva_file                = "${projectDir}/test/gsva_sample.csv"
params.cell_file_graphdrp       = "${projectDir}/test/mu_sample.csv"

// --- Environment & Dependencies ---
params.local_wheels_cp37 = "${projectDir}/local_wheels/cp37"
params.local_wheels_cp38 = "${projectDir}/local_wheels/cp38"

// --- Ensemble Script Inputs ---
params.model_pkl = "${projectDir}/best_model_RandomForest_rmse.pkl"
// 【核心修复 1】: weight_file 参数已不再需要，将其注释或删除
// params.weight_file = "${projectDir}/weight.csv"


log.info """
===================================================
DRP Main Workflow Started (Optimized Model Set)
===================================================
Entry Point          : ${params.entry}
Output Directory     : ${params.output_dir}
--- GPU Allocation ---
Part1 Models         : GPU ${params.gpu_part1}
DIPK & GraphDRP      : GPU ${params.gpu_dipk_graphdrp}
--- Core Input Files ---
Drug SMILES          : ${params.drug_smiles}
Gene Expression      : ${params.gene_exp_file}
Mutation             : ${params.mutation_file}
===================================================
""".stripIndent()

// ========================================================================================
//                     Include Sub-workflows and Processes
// ========================================================================================
include { run_part1_workflow } from './workflows/nf_part1.nf'
include { run_dipk_graphdrp_workflow } from './workflows/nf_DIPK-GraphDRP.nf'
include { run_ensemble as run_voter } from './workflows/voter.nf'

// ========================================================================================
//                                Main Workflow
// ========================================================================================
workflow {
    file(params.output_dir).mkdirs()

    if (params.entry == 'voter') {
        def prediction_files = Channel.fromPath("${params.output_dir}/*_predictions*.csv").collect()

        // 【核心修复 2】: 调用 run_voter 时，不再传递 weight_file，现在只需要3个参数
        run_voter(
            prediction_files,
            params.output_dir,
            file(params.model_pkl)
        )
    }
    else if (params.entry == 'dipk_graphdrp') {
        // --- 这部分保持你的原样，不做改动 ---
        def dipk_graphdrp_input_params = [
            drug_smiles: file(params.drug_smiles),
            gene_exp_file: file(params.gene_exp_file),
            cell_file_graphdrp: file(params.cell_file_graphdrp)
        ]

        run_dipk_graphdrp_workflow(
            params.gpu_dipk_graphdrp,
            file(params.local_wheels_cp38),
            params.output_dir,
            dipk_graphdrp_input_params
        )
    }
    else if (params.entry == 'part1') {
        // --- 这部分保持你的原样，不做改动 ---
        def part1_input_params = [
            drug_smiles: file(params.drug_smiles),
            gene_exp_file: file(params.gene_exp_file),
            mutation_file: file(params.mutation_file),
            cnv_file: file(params.cnv_file),
            gsva_file: file(params.gsva_file)
        ]
        
        run_part1_workflow(
            params.gpu_part1,
            file(params.local_wheels_cp37),
            params.output_dir,
            part1_input_params
        )
    }
    else {
        log.error """
        ERROR: No valid workflow entry point specified. Please use --entry.
        Available: 'voter', 'dipk_graphdrp', 'part1'
        """.stripIndent()
        exit 1
    }
}

// ========================================================================================
//                            Workflow Completion Callback
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