#!/bin/bash
set -euo pipefail

BASE_DIR=$(pwd)
DRUG_SMILES="${BASE_DIR}/test/drug_sample.csv"
GENE_EXP="${BASE_DIR}/test/gene_sample.csv"
MUTATION="${BASE_DIR}/test/mu_sample.csv"
CNV="${BASE_DIR}/test/cnv_sample.csv"
MICRORNA="${BASE_DIR}/test/mi_sample.csv"
GSVA="${BASE_DIR}/test/gsva_sample.csv"
CELL_GRAPHDRP="${BASE_DIR}/test/mu_sample.csv"

FINAL_OUTPUT_DIR="${BASE_DIR}/results_final"
SMART_WS="${BASE_DIR}/smart_workspace"

format_time() {
    local T=$1
    local H=$((T/3600))
    local M=$((T%3600/60))
    local S=$((T%60))
    printf "%02d:%02d:%02d" $H $M $S
}

cleanup_nextflow() {
    echo "INFO: Cleaning work directory..."
    if [ -d "work" ]; then
        find work -maxdepth 1 ! -name 'conda' ! -name 'work' -exec rm -rf {} +
    fi
    rm -rf .nextflow .nextflow.log*
}

run_parallel_stage() {
    local entry=$1
    local total=$2
    local max_p=$3
    local gpus=$4
    
    local stage_start_time=$(date +%s)
    local total_batches=$(( (total + max_p - 1) / max_p ))
    local current_batch=0

    echo ">>> Starting Stage: $entry | Total: $total chunks | Batches: $total_batches"
    
    for i in $(seq 0 $((total-1))); do
        if [ $((i % max_p)) -eq 0 ]; then
            current_batch=$((current_batch + 1))
            batch_start_time=$(date +%s)
        fi

        CHUNK_ID=$(printf "%03d" $i)
        CHUNK_OUT_DIR="${SMART_WS}/results/${entry}_${CHUNK_ID}"
        mkdir -p "$CHUNK_OUT_DIR"
        TARGET_GPU_ID=$(( i % gpus ))

        echo "      Launching Chunk $CHUNK_ID on GPU $TARGET_GPU_ID..."
        
        nextflow run main.nf \
            --entry "$entry" \
            --drug_smiles "$(readlink -f ${SMART_WS}/plan/chunk_${CHUNK_ID}.csv)" \
            --gene_exp_file "$GENE_EXP" \
            --mutation_file "$MUTATION" \
            --cnv_file "$CNV" \
            --microrna_file "$MICRORNA" \
            --gsva_file "$GSVA" \
            --cell_file_graphdrp "$CELL_GRAPHDRP" \
            --gpu_map.GPDRP $TARGET_GPU_ID \
            --gpu_map.BANDRP $TARGET_GPU_ID \
            --gpu_map.DeepTTC $TARGET_GPU_ID \
            --gpu_map.paccmann $TARGET_GPU_ID \
            --gpu_map.Precily $TARGET_GPU_ID \
            --gpu_map.DIPK $TARGET_GPU_ID \
            --gpu_map.GraphDRP $TARGET_GPU_ID \
            --output_dir "$CHUNK_OUT_DIR" \
            -work-dir "${BASE_DIR}/work/work_${entry}_${CHUNK_ID}" &

        if [ $(( (i + 1) % max_p )) -eq 0 ] || [ $((i + 1)) -eq $total ]; then
            wait
            batch_end_time=$(date +%s)
            
            local batch_dur=$((batch_end_time - batch_start_time))
            local stage_elap=$((batch_end_time - stage_start_time))
            local b_left=$((total_batches - current_batch))
            local eta=$((b_left * batch_dur))
            
            echo "---------------------------------------------------"
            echo " Stage Progress: Batch $current_batch/$total_batches"
            echo " Last Batch Cost: $(format_time $batch_dur) | Stage Elapsed: $(format_time $stage_elap)"
            [ $b_left -gt 0 ] && echo " Est. Stage Remaining: $(format_time $eta)"
            echo "---------------------------------------------------"
            cleanup_nextflow
        fi
    done
}

# ===================================================
#                    Workflow
# ===================================================
GLOBAL_START=$(date +%s)
echo "==================================================="
echo "  Smart Multi-Omics Ensemble Workflow V2.3"
echo "==================================================="

cleanup_nextflow
rm -rf "$SMART_WS" "$FINAL_OUTPUT_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"

echo "--- [0/3] Planning & Data Splitting ---"

nextflow run smart.nf --drug_file "$DRUG_SMILES" --cell_file "$GSVA" --output_dir "$SMART_WS"

if [ ! -f "${SMART_WS}/plan/config.sh" ]; then
    echo "ERROR: smart.nf failed to generate config.sh"
    exit 1
fi
source "${SMART_WS}/plan/config.sh"

stage1_start=$(date +%s)
run_parallel_stage "dipk_graphdrp" "$TOTAL_CHUNKS" "$MAX_PARALLEL" "$GPU_COUNT"
stage1_end=$(date +%s)

stage2_start=$(date +%s)
run_parallel_stage "part1" "$TOTAL_CHUNKS" "$MAX_PARALLEL" "$GPU_COUNT"
stage2_end=$(date +%s)

echo "--- [POST] Consolidating Predicted Fragments ---"
python3 -c "
import pandas as pd
import glob
import os

results_path = '${SMART_WS}/results'
final_path = '${FINAL_OUTPUT_DIR}'

all_csv_paths = glob.glob(os.path.join(results_path, '*', '*.csv'))
if not all_csv_paths:
    print('ERROR: No CSV files found in ' + results_path)
else:
    file_names = set([os.path.basename(f) for f in all_csv_paths])
    for name in file_names:
        chunks = sorted(glob.glob(os.path.join(results_path, '*', name)))
        print(f'   - Merging {len(chunks)} chunks for {name}')
        df_list = [pd.read_csv(f) for f in chunks if os.path.getsize(f) > 10]
        if df_list:
            combined_df = pd.concat(df_list).drop_duplicates()
            combined_df.to_csv(os.path.join(final_path, name), index=False)
"

echo "--- [3/3] Ensemble Stage (Voter) ---"

nextflow run main.nf --entry voter --output_dir "$FINAL_OUTPUT_DIR"


GLOBAL_END=$(date +%s)
echo "==================================================="
echo "  Execution Summary"
echo "==================================================="
echo " DIPK Stage Duration    : $(format_time $((stage1_end - stage1_start)))"
echo " Part1 Stage Duration   : $(format_time $((stage2_end - stage2_start)))"
echo " Total Elapsed Time     : $(format_time $((GLOBAL_END - GLOBAL_START)))"
echo " Final Output Directory : $FINAL_OUTPUT_DIR"
echo "==================================================="
