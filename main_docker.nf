#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
========================================================================================
    DRP Docker Unified Workflow 
========================================================================================
*/

// ========================================================================================
//                             1. Global parameter definition
// ========================================================================================

params.entry        = 'part1' 
params.output_dir   = "/app/results"
params.model_pkl    = "/app/best_model_RandomForest_rmse.pkl"

// --- GPU ---
params.gpu_map = [
    GPDRP:    0,
    BANDRP:   0,
    DeepTTC:  0,
    paccmann: 0,
    Precily:  0,
    DIPK:     0,
    GraphDRP: 0
]

// --- Drug ---
params.drug_list_option   = 0
params.drug_types         = 'ALL'
params.test_mode_limit    = 0
params.predict_chemo_file = "/app/test/predict_chemo.csv"
params.predict_np_file    = "/app/test/predict_np.csv"

// --- Input ---
params.drug_smiles        = "/app/test/drug_sample.csv"
params.mutation_file      = "/app/test/mu_sample.csv"
params.gene_exp_file      = "/app/test/gene_sample.csv"
params.cnv_file           = "/app/test/cnv_sample.csv"
params.gsva_file          = "/app/test/gsva_sample.csv"
params.cell_file_graphdrp = "/app/test/mu_sample.csv"

// --- env ---
def PY37_PYTHON = "/opt/conda/envs/part1_env/bin/python"
def PY37_LIB    = "/opt/conda/envs/part1_env/lib"
def PY38_PYTHON = "/opt/conda/envs/dipk_graphdrp_env/bin/python"
def PY38_LIB    = "/opt/conda/envs/dipk_graphdrp_env/lib"

log.info """
===================================================
DRP Unified Workflow - Docker Integrated Mode
===================================================
Entry Point      : ${params.entry}
Output Directory : ${params.output_dir}
GPU Mapping      : ${params.gpu_map}
===================================================
""".stripIndent()

// ====================================================================================
// 2. drug list
// ====================================================================================
process PREPARE_DRUG_LIST {
    tag "Option ${drug_option}"
    publishDir "${params.output_dir}/input_files", mode: 'copy', overwrite: true
    input:
    val drug_option; val drug_types; val test_limit
    path chemo_file; path np_file; path source_file
    output: path "prepared_drug_list.csv"
    script:
    def s_abs = source_file.toRealPath()
    def c_abs = chemo_file.toRealPath()
    def n_abs = np_file.toRealPath()
    """
    #!/bin/bash
    set -e
    echo "[VAR_CHECK] Option: ${drug_option}, Source: ${s_abs}"

    DRUG_OPTION="${drug_option}"
    DRUG_TYPES="${drug_types}"
    TEST_LIMIT="${test_limit}"
    CHEMO_FILE="${c_abs}"
    NP_FILE="${n_abs}"
    SOURCE_FILE="${s_abs}"

    map_chemo_alias_to_phase() { case \$1 in PR) echo "Preclinical" ;; P2) echo "Phase 2" ;; AP) echo "Approved" ;; P3) echo "Phase 3" ;; P1) echo "Phase 1" ;; E1) echo "Early Phase 1" ;; *) echo "" ;; esac; }
    map_np_alias_to_pathway() { local sorted_alias=\$(echo "\$1" | tr '[:upper:]' '[:lower:]' | grep -o . | sort | tr -d '\\n'); case \$sorted_alias in all) echo "__ALL_ENTRIES__" ;; a) echo "Shikimates and Phenylpropanoids" ;; b) echo "Polyketides" ;; c) echo "Terpenoids" ;; d) echo "Amino acids and Peptides" ;; e) echo "Carbohydrates" ;; f) echo "Alkaloids" ;; g) echo "Fatty acids" ;; ab) echo "Polyketides, Shikimates and Phenylpropanoids" ;; ac) echo "Shikimates and Phenylpropanoids, Terpenoids" ;; ad) echo "Amino acids and Peptides, Shikimates and Phenylpropanoids" ;; ae) echo "Carbohydrates, Shikimates and Phenylpropanoids" ;; af) echo "Alkaloids, Shikimates and Phenylpropanoids" ;; ag) echo "Fatty acids, Shikimates and Phenylpropanoids" ;; bc) echo "Polyketides, Terpenoids" ;; bd) echo "Amino acids and Peptides, Polyketides" ;; be) echo "Carbohydrates, Polyketides" ;; bf) echo "Alkaloids, Polyketides" ;; bg) echo "Fatty acids, Polyketides" ;; cd) echo "Amino acids and Peptides, Terpenoids" ;; ce) echo "Carbohydrates, Terpenoids" ;; cf) echo "Alkaloids, Terpenoids" ;; cg) echo "Fatty acids, Terpenoids" ;; de) echo "Amino acids and Peptides, Carbohydrates" ;; df) echo "Alkaloids, Amino acids and Peptides" ;; dg) echo "Amino acids and Peptides, Fatty acids" ;; ef) echo "Alkaloids, Carbohydrates" ;; eg) echo "Carbohydrates, Fatty acids" ;; fg) echo "Alkaloids, Fatty acids" ;; abc) echo "Polyketides, Shikimates and Phenylpropanoids, Terpenoids" ;; bcf) echo "Alkaloids, Polyketides, Terpenoids" ;; bcg) echo "Fatty acids, Polyketides, Terpenoids" ;; bdf) echo "Alkaloids, Amino acids and Peptides, Polyketides" ;; bdg) echo "Amino acids and Peptides, Fatty acids, Polyketides" ;; cdf) echo "Alkaloids, Amino acids and Peptides, Terpenoids" ;; adfg) echo "Alkaloids, Amino acids and Peptides, Fatty acids" ;; adf) echo "Alkaloids, Amino acids and Peptides, Shikimates and Phenylpropanoids" ;; acd) echo "Amino acids and Peptides, Shikimates and Phenylpropanoids, Terpenoids" ;; ade) echo "Amino acids and Peptides, Carbohydrates, Shikimates and Phenylpropanoids" ;; *) echo "" ;; esac; }
    
    LIMIT_CMD=""; if [[ "\$TEST_LIMIT" -gt 0 ]]; then LIMIT_CMD="| head -n \${TEST_LIMIT}"; fi
    echo "drug_name,smiles" > header.csv
    
    if [ "\${DRUG_OPTION}" == "0" ] || [ "\${DRUG_OPTION}" == "3" ]; then
        cat header.csv <(sed '1d' "\${SOURCE_FILE}" 2>/dev/null || cat "\${SOURCE_FILE}") | eval "cat \${LIMIT_CMD}" > prepared_drug_list.csv
    elif [ "\${DRUG_OPTION}" == "1" ]; then
        awk_condition=""; if [[ "\${DRUG_TYPES^^}" == "ALL" ]]; then awk_condition="(\\\$3 ~ /Preclinical|Phase 2|Approved|Phase 3|Phase 1|Early Phase 1/)"; else IFS=',' read -ra TYPES <<< "\${DRUG_TYPES^^}"; for type in "\${TYPES[@]}"; do phase=\$(map_chemo_alias_to_phase "\$type"); if [ -n "\$phase" ]; then [ -n "\$awk_condition" ] && awk_condition+=" || "; awk_condition+="\\\$3 ~ /"\$phase"/"; fi; done; fi
        cat header.csv <(sed 's/\\r\$//' "\${CHEMO_FILE}" | awk -F',' "BEGIN{OFS=\\",\\"} (\$awk_condition) {print \\\$1, \\\$2}" | sort -u | eval "cat \${LIMIT_CMD}") > prepared_drug_list.csv
    elif [ "\${DRUG_OPTION}" == "2" ]; then
        awk_condition=""; if [[ "\${DRUG_TYPES^^}" == "ALL" ]]; then awk_condition="NF >= 3"; else IFS=',' read -ra TYPES <<< "\${DRUG_TYPES}"; for type in "\${TYPES[@]}"; do pathway=\$(map_np_alias_to_pathway "\$type"); if [[ "\$pathway" == "__ALL_ENTRIES__" ]]; then awk_condition="NF >= 3"; break; fi; if [ -z "\$pathway" ]; then pathway="\$type"; fi; [ -n "\$awk_condition" ] && awk_condition+=" || "; awk_condition+="\\\$3 ~ /"\$pathway"/"; done; fi
        cat header.csv <(sed 's/\\r\$//' "\${NP_FILE}" | awk -F',' "BEGIN{OFS=\\",\\"} (\$awk_condition) {print \\\$1, \\\$2}" | sort -u | eval "cat \${LIMIT_CMD}") > prepared_drug_list.csv
    fi
    """
}

// ====================================================================================
// 3. Part 1 (Python 3.7)
// ====================================================================================
process run_BANDRP {
    tag "BANDRP"; conda "/opt/conda/envs/part1_env"
    input: val gpu_id; path drug_smiles; path gene_exp; path mut; path cnv
    script:
    def d_abs = drug_smiles.toRealPath(); def g_abs = gene_exp.toRealPath()
    def m_abs = mut.toRealPath(); def c_abs = cnv.toRealPath()
    """
    export LD_LIBRARY_PATH=${PY37_LIB}:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd /app/BANDRP-main
    ${PY37_PYTHON} ./predict_all.py --model_path ./github_upload/output_dir/db1/model.pt \
        --new_drugs_csv "${d_abs}" --exp_path "${g_abs}" --mut_path "${m_abs}" --cnv_path "${c_abs}" \
        --output_csv "${params.output_dir}/bandrp_predictions.csv" --cuda_id ${gpu_id}
    """
}

process run_DeepTTC {
    tag "DeepTTC"; conda "/opt/conda/envs/part1_env"
    input: val gpu_id; path drug_smiles; path gene_exp
    script:
    def d_abs = drug_smiles.toRealPath(); def g_abs = gene_exp.toRealPath()
    """
    export LD_LIBRARY_PATH=${PY37_LIB}:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    cd /app/DeepTTC
    ${PY37_PYTHON} ./predict_all.py --new_drug_file "${d_abs}" --new_cell_line_file "${g_abs}" \
        --training_gene_list_file "./mydata/expt.txt" --model_dir "./DeepTTC" --vocab_dir "./ESPF" \
        --output_file "${params.output_dir}/DeepTTC_predictions.csv"
    """
}

process run_GPDRP {
    tag "GPDRP"; conda "/opt/conda/envs/part1_env"
    input: val gpu_id; val model_type; val model_filename; path drug_smiles; path gsva
    script:
    def d_abs = drug_smiles.toRealPath(); def g_abs = gsva.toRealPath()
    """
    export LD_LIBRARY_PATH=${PY37_LIB}:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd /app/GPDRP
    ${PY37_PYTHON} ./predict_all.py --smiles_file "${d_abs}" --new_cell_line_file "${g_abs}" \
        --training_gene_expression_file "./mydata/exp.txt" --model_file "./output/models/${model_filename}" \
        --output_file "${params.output_dir}/GPDRP_predictions_${model_type}.csv" --model_type "${model_type}" \
        --cuda_name "cuda:${gpu_id}"
    """
}

process run_paccmann {
    tag "paccmann"; conda "/opt/conda/envs/part1_env"
    input: val gpu_id; path drug_smiles; path gene_exp
    script:
    def d_abs = drug_smiles.toRealPath(); def g_abs = gene_exp.toRealPath()
    """
    export LD_LIBRARY_PATH=${PY37_LIB}:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    cd /app/paccmann_predictor-master
    ${PY37_PYTHON} ./predict_all.py --predict_smiles_filepath "${d_abs}" --gep_filepath "${g_abs}" \
        --model_run_path ./paccman_training_runs/paccmann_train_1747977832 \
        --gene_filepath_spec ./data/2128_genes.pkl --smiles_language_filepath ./paccmann/smiles_language.pkl \
        --output_filepath "${params.output_dir}/paccmann_predictions.csv"
    """
}

process run_Precily {
    tag "Precily"; conda "/opt/conda/envs/part1_env"
    input: val gpu_id; path drug_smiles; path gsva
    script:
    def d_abs = drug_smiles.toRealPath(); def g_abs = gsva.toRealPath()
    """
    export LD_LIBRARY_PATH=${PY37_LIB}:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd /app/Precily-v1.0.0/Pathway_based
    ${PY37_PYTHON} ./predict_all.py --input_drugs_file "${d_abs}" --input_cell_lines_file "${g_abs}" \
        --models_dir . --output_file "${params.output_dir}/Precily_predictions.csv" \
        --drug_embedding_file ../mydata/utils/drug.pubchem.canon.l8.ws20.txt \
        --elements_file ../mydata/utils/elements.txt --cell_features_template_file ../mydata/mycell_gsva2.csv
    """
}

// ====================================================================================
// 4. DIPK & GraphDRP (Python 3.8)
// ====================================================================================
process run_GraphDRP {
    tag "GraphDRP"; conda "/opt/conda/envs/dipk_graphdrp_env"
    input: val gpu_id; val model_type; val model_filename; path drug_smiles; path cell_file
    script:
    def d_abs = drug_smiles.toRealPath(); def c_abs = cell_file.toRealPath()
    """
    export LD_LIBRARY_PATH=${PY38_LIB}:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd /app/GraphDRP-master
    ${PY38_PYTHON} ./predict_all.py --drug_file "${d_abs}" --cell_file "${c_abs}" \
        --model_path "./${model_filename}" --model_type "${model_type}" --data_dir ./mydata/ \
        --output_file "${params.output_dir}/GraphDRP_predictions_${model_type}.csv" --cuda_name "cuda:${gpu_id}"
    """
}

process run_DIPK {
    tag "DIPK"; conda "/opt/conda/envs/dipk_graphdrp_env"
    input: val gpu_id; path drug_smiles; path gene_exp
    script:
    def d_abs = drug_smiles.toRealPath(); def g_abs = gene_exp.toRealPath()
    """
    export LD_LIBRARY_PATH=${PY38_LIB}:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    cd /app/DIPK-main/prog
    ${PY38_PYTHON} ./predict_all.py \
        --input_drugs_csv "${d_abs}" --gene_expression_file "${g_abs}" \
        --output_csv "${params.output_dir}/DIPK_predictions.csv" --model_path ./result/Train.pkl \
        --train_config_path ./TrainConfig.py --data_config_path ./DataConfig.py \
        --molgnet_model_path ./Data/MolGNet.pt --bionic_dict_path ../Dataset/BIONIC_dict.pkl \
        --canonical_gene_list_path ../Dataset/exp.txt --device cuda
    """
}

// ====================================================================================
// 5. Ensemble Voter 
// ====================================================================================
process run_VOTER {
    tag "Voter"
    script:
    """
    echo "[VAR_CHECK] Voter -> Running in ${params.output_dir}"
    cd ${params.output_dir}
    ${PY37_PYTHON} /app/voter.py --mode 1 --model_pkl "${params.model_pkl}"
    """
}

// ====================================================================================
// 6.workflow
// ====================================================================================
workflow {
    def source_drug_file = (params.drug_list_option == 3) ? file("/app/test/drug_sample.csv") : file(params.drug_smiles)
    ch_prepared_drug_list = PREPARE_DRUG_LIST(params.drug_list_option, params.drug_types, params.test_mode_limit, \
        file(params.predict_chemo_file), file(params.predict_np_file), source_drug_file)
    
    ch_validated_drug_list = ch_prepared_drug_list.map { f ->
        if (f.readLines().size() <= 1) { error "Drug list is empty." } else { return f }
    }

    if (params.entry == 'part1') {
        run_BANDRP(params.gpu_map.BANDRP, ch_validated_drug_list, file(params.gene_exp_file), file(params.mutation_file), file(params.cnv_file))
        run_DeepTTC(params.gpu_map.DeepTTC, ch_validated_drug_list, file(params.gene_exp_file))
        run_paccmann(params.gpu_map.paccmann, ch_validated_drug_list, file(params.gene_exp_file))
        run_Precily(params.gpu_map.Precily, ch_validated_drug_list, file(params.gsva_file))
        gpdrp_models = Channel.fromList([['GAT', 'model_GAT_GDSC_drug_blind_run2.model'], ['GCN', 'model_GCN_GDSC_drug_blind_run2.model']])
        run_GPDRP(params.gpu_map.GPDRP, gpdrp_models.map{it[0]}, gpdrp_models.map{it[1]}, ch_validated_drug_list, file(params.gsva_file))
    }
    else if (params.entry == 'dipk_graphdrp') {
        run_DIPK(params.gpu_map.DIPK, ch_validated_drug_list, file(params.gene_exp_file))
        graphdrp_models = Channel.fromList([['GATNet', 'model_GATNet_GDSC_blind_run1.model'], ['GAT_GCN', 'model_GAT_GCN_GDSC_blind_run1.model']])
        run_GraphDRP(params.gpu_map.GraphDRP, graphdrp_models.map{it[0]}, graphdrp_models.map{it[1]}, ch_validated_drug_list, file(params.cell_file_graphdrp))
    }
    else if (params.entry == 'voter') {
        run_VOTER()
    }
}
