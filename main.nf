#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    Main Routing Workflow
========================================================================================
*/

// ========================================================================================
//                             Top-Level Global Parameters
// ========================================================================================

// --- Workflow Control & Environment ---
params.entry = ''
params.output_dir = "${projectDir}/results"

params.gpu_map = [
    // Part 1 Models
    GPDRP:    0,
    BANDRP:   0,
    DeepTTC:  0,
    paccmann: 0,
    Precily:  0,
    // DIPK & GraphDRP Models
    DIPK:     0,
    GraphDRP: 0
]

// --- Drug List Selection Parameters ---
params.drug_list_option = 0
params.drug_types       = 'ALL'
params.test_mode_limit  = 0
params.predict_chemo_file = "${projectDir}/test/predict_chemo.csv"
params.predict_np_file    = "${projectDir}/test/predict_np.csv"

// --- User Input File Paths ---
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

log.info """
===================================================
DRP Main Workflow Started (Optimized Model Set)
===================================================
Entry Point          : ${params.entry}
Output Directory     : ${params.output_dir}
--- GPU Allocation ---
GPDRP    : GPU ${params.gpu_map.GPDRP}
BANDRP   : GPU ${params.gpu_map.BANDRP}
DeepTTC  : GPU ${params.gpu_map.DeepTTC}
paccmann : GPU ${params.gpu_map.paccmann}
Precily  : GPU ${params.gpu_map.Precily}
DIPK     : GPU ${params.gpu_map.DIPK}
GraphDRP : GPU ${params.gpu_map.GraphDRP}
===================================================
""".stripIndent()

// ========================================================================================
//                     Include Sub-workflows
// ========================================================================================
include { run_part1_workflow } from './workflows/nf_part1.nf';
include { run_dipk_graphdrp_workflow } from './workflows/nf_DIPK-GraphDRP.nf';
include { run_ensemble as run_voter } from './workflows/voter.nf';

// ====================================================================================
//    PREPARE_DRUG_LIST Process
// ====================================================================================
process PREPARE_DRUG_LIST {
    tag "Option ${drug_option}"
    publishDir "${params.output_dir}/input_files", mode: 'copy', overwrite: true
    input:
    val drug_option; val drug_types; val test_limit
    path chemo_file; path np_file; path source_file
    output: path "prepared_drug_list.csv"
    script:
    """
    #!/bin/bash
    set -e; DRUG_OPTION="${drug_option}"; DRUG_TYPES="${drug_types}"; TEST_LIMIT="${test_limit}"; CHEMO_FILE="${chemo_file}"; NP_FILE="${np_file}"; SOURCE_FILE="${source_file}";
    map_chemo_alias_to_phase() { case \$1 in PR) echo "Preclinical" ;; P2) echo "Phase 2" ;; AP) echo "Approved" ;; P3) echo "Phase 3" ;; P1) echo "Phase 1" ;; E1) echo "Early Phase 1" ;; *) echo "" ;; esac; }
    map_np_alias_to_pathway() { local sorted_alias=\$(echo "\$1" | tr '[:upper:]' '[:lower:]' | grep -o . | sort | tr -d '\\n'); case \$sorted_alias in all) echo "__ALL_ENTRIES__" ;; a) echo "Shikimates and Phenylpropanoids" ;; b) echo "Polyketides" ;; c) echo "Terpenoids" ;; d) echo "Amino acids and Peptides" ;; e) echo "Carbohydrates" ;; f) echo "Alkaloids" ;; g) echo "Fatty acids" ;; ab) echo "Polyketides, Shikimates and Phenylpropanoids" ;; ac) echo "Shikimates and Phenylpropanoids, Terpenoids" ;; ad) echo "Amino acids and Peptides, Shikimates and Phenylpropanoids" ;; ae) echo "Carbohydrates, Shikimates and Phenylpropanoids" ;; af) echo "Alkaloids, Shikimates and Phenylpropanoids" ;; ag) echo "Fatty acids, Shikimates and Phenylpropanoids" ;; bc) echo "Polyketides, Terpenoids" ;; bd) echo "Amino acids and Peptides, Polyketides" ;; be) echo "Carbohydrates, Polyketides" ;; bf) echo "Alkaloids, Polyketides" ;; bg) echo "Fatty acids, Polyketides" ;; cd) echo "Amino acids and Peptides, Terpenoids" ;; ce) echo "Carbohydrates, Terpenoids" ;; cf) echo "Alkaloids, Terpenoids" ;; cg) echo "Fatty acids, Terpenoids" ;; de) echo "Amino acids and Peptides, Carbohydrates" ;; df) echo "Alkaloids, Amino acids and Peptides" ;; dg) echo "Amino acids and Peptides, Fatty acids" ;; ef) echo "Alkaloids, Carbohydrates" ;; eg) echo "Carbohydrates, Fatty acids" ;; fg) echo "Alkaloids, Fatty acids" ;; abc) echo "Polyketides, Shikimates and Phenylpropanoids, Terpenoids" ;; bcf) echo "Alkaloids, Polyketides, Terpenoids" ;; bcg) echo "Fatty acids, Polyketides, Terpenoids" ;; bdf) echo "Alkaloids, Amino acids and Peptides, Polyketides" ;; bdg) echo "Amino acids and Peptides, Fatty acids, Polyketides" ;; cdf) echo "Alkaloids, Amino acids and Peptides, Terpenoids" ;; adfg) echo "Alkaloids, Amino acids and Peptides, Fatty acids" ;; adf) echo "Alkaloids, Amino acids and Peptides, Shikimates and Phenylpropanoids" ;; acd) echo "Amino acids and Peptides, Shikimates and Phenylpropanoids, Terpenoids" ;; ade) echo "Amino acids and Peptides, Carbohydrates, Shikimates and Phenylpropanoids" ;; *) echo "" ;; esac; }
    LIMIT_CMD=""; if [[ "\$TEST_LIMIT" -gt 0 ]]; then LIMIT_CMD="| head -n \${TEST_LIMIT}"; fi
    echo "drug_name,smiles" > header.csv
    if [ "\${DRUG_OPTION}" == "0" ] || [ "\${DRUG_OPTION}" == "3" ]; then
        cat header.csv <(sed '1d' "\${SOURCE_FILE}" 2>/dev/null || cat "\${SOURCE_FILE}") | eval "cat \${LIMIT_CMD}" > prepared_drug_list.csv
    elif [ "\${DRUG_OPTION}" == "1" ]; then
        if [ ! -r "\$CHEMO_FILE" ]; then echo "FATAL ERROR..." >&2; exit 1; fi
        awk_condition=""; if [[ "\${DRUG_TYPES^^}" == "ALL" ]]; then awk_condition="(\\\$3 ~ /Preclinical|Phase 2|Approved|Phase 3|Phase 1|Early Phase 1/)"; else IFS=',' read -ra TYPES <<< "\${DRUG_TYPES^^}"; for type in "\${TYPES[@]}"; do phase=\$(map_chemo_alias_to_phase "\$type"); if [ -n "\$phase" ]; then [ -n "\$awk_condition" ] && awk_condition+=" || "; awk_condition+="\\\$3 ~ /"\$phase"/"; fi; done; fi
        if [ -z "\$awk_condition" ]; then echo "ERROR..." >&2; exit 1; fi
        cat header.csv <(sed 's/\\r\$//' "\${CHEMO_FILE}" | awk -F',' "BEGIN{OFS=\\",\\"} (\$awk_condition) {print \\\$1, \\\$2}" | sort -u | eval "cat \${LIMIT_CMD}") > prepared_drug_list.csv
    elif [ "\${DRUG_OPTION}" == "2" ]; then
        if [ ! -r "\$NP_FILE" ]; then echo "FATAL ERROR..." >&2; exit 1; fi
        awk_condition=""; if [[ "\${DRUG_TYPES^^}" == "ALL" ]]; then awk_condition="NF >= 3"; else IFS=',' read -ra TYPES <<< "\${DRUG_TYPES}"; for type in "\${TYPES[@]}"; do pathway=\$(map_np_alias_to_pathway "\$type"); if [[ "\$pathway" == "__ALL_ENTRIES__" ]]; then awk_condition="NF >= 3"; break; fi; if [ -z "\$pathway" ]; then pathway="\$type"; fi; [ -n "\$awk_condition" ] && awk_condition+=" || "; awk_condition+="\\\$3 ~ /"\$pathway"/"; done; fi
        if [ -z "\$awk_condition" ]; then echo "ERROR..." >&2; exit 1; fi
        cat header.csv <(sed 's/\\r\$//' "\${NP_FILE}" | awk -F',' "BEGIN{OFS=\\",\\"} \${awk_condition) {print \\\$1, \\\$2}" | sort -u | eval "cat \${LIMIT_CMD}") > prepared_drug_list.csv
    else
        echo "ERROR..." >&2; exit 1
    fi
    if [ \$(wc -l < prepared_drug_list.csv) -gt 1 ]; then echo "OK"; else echo "WARNING: Empty"; fi
    """
}

// ========================================================================================
//                                Main Workflow
// ========================================================================================
workflow {
    file(params.output_dir).mkdirs()
    def source_drug_file
    if (params.drug_list_option == 3) { source_drug_file = file("${projectDir}/test/drug_sample.csv") }
    else { source_drug_file = file(params.drug_smiles) }

    ch_prepared_drug_list = PREPARE_DRUG_LIST(params.drug_list_option, params.drug_types, params.test_mode_limit, file(params.predict_chemo_file), file(params.predict_np_file), source_drug_file)
    ch_validated_drug_list = ch_prepared_drug_list.map { file ->
        if (file.readLines().size() <= 1) { error "STOPPING: The generated drug list is empty." }
        else { return file }
    }

    if (params.entry == 'voter') {
        run_voter(Channel.fromPath("${params.output_dir}/*_predictions*.csv").collect(), params.output_dir, file(params.model_pkl))
    }
    else if (params.entry == 'dipk_graphdrp') {
        ch_inputs = ch_validated_drug_list.map { f -> [drug_smiles: f, gene_exp_file: file(params.gene_exp_file), cell_file_graphdrp: file(params.cell_file_graphdrp)] }
        run_dipk_graphdrp_workflow(
            params.gpu_map, // Pass the entire GPU Map
            file(params.local_wheels_cp38),
            params.output_dir,
            ch_inputs
        )
    }
    else if (params.entry == 'part1') {
        ch_inputs = ch_validated_drug_list.map { f -> [drug_smiles: f, gene_exp_file: file(params.gene_exp_file), mutation_file: file(params.mutation_file), cnv_file: file(params.cnv_file), gsva_file: file(params.gsva_file)] }
        run_part1_workflow(
            params.gpu_map, // Pass the entire GPU Map
            file(params.local_wheels_cp37),
            params.output_dir,
            ch_inputs
        )
    }
    else {
        error "ERROR: No valid workflow entry point specified."
    }
}

// ========================================================================================
//                            Workflow Completion Callback
// ========================================================================================
workflow.onComplete {
    if (workflow.success) { log.info "Workflow Completed Successfully!" }
    else { log.error "Workflow FAILED! Error: ${workflow.errorReport}" }
}