#!/bin/bash
#
# ==============================================================================
#   DRP Ensemble Workflow - Full Sequential Run
# ==============================================================================
#
#   此脚本按顺序执行所有的模型预测流程，并在所有模型成功运行后，
#   最后调用voter流程进行集成投票。
#
#   特性:
#   - 顺序执行: 如果任何一步失败，脚本将立即退出。
#   - 参数传递: 您可以向此脚本传递任何Nextflow参数(如 --gene_exp_file)，
#               这些参数将被自动应用于所有需要它们的模型流程。
#   - 统一输出: 所有结果将保存在一个以时间戳命名的目录中。
#
#   用法:
#   1. 使用默认测试数据:
#      ./run_ensemble.sh
#
#   2. 使用您自己的数据并分配GPU:
#      ./run_ensemble.sh --gene_exp_file /path/to/exp.csv \
#                        --mutation_file /path/to/mut.csv \
#                        --gpu_part1 0 --gpu_deepcdr 1
#
# ==============================================================================

# --- 脚本设置 ---
# set -e: 如果任何命令以非零状态退出，则立即退出脚本。
# set -u: 将未设置的变量视为错误。
# set -o pipefail: 如果管道中的任何命令失败，则整个管道的退出状态为失败。
set -euo pipefail

# --- 定义统一的输出目录 ---
# 创建一个带有日期和时间戳的唯一目录名，以防覆盖之前的结果。
OUTPUT_DIR="ensemble_results_$(date +%Y%m%d_%H%M%S)"
echo "==================================================="
echo "  Starting Full Ensemble Prediction Workflow"
echo "==================================================="
echo
echo "INFO: All prediction results will be stored in: $OUTPUT_DIR"
echo "INFO: Additional arguments passed to Nextflow: $@"
echo

# ========================= 步骤 1: 运行 Part 1 模型组 =========================
echo "--- [1/5] Running Part 1 Models (BANDRP, DeepTTC, GADRP, etc.) ---"
nextflow run main.nf -profile part1 --output_dir "$OUTPUT_DIR" "$@"
echo "--- Part 1 Models COMPLETE ---"
echo

# ========================= 步骤 2: 运行 DeepAEG 模型 =========================
echo "--- [2/5] Running DeepAEG Model ---"
nextflow run main.nf -profile deepaeg --output_dir "$OUTPUT_DIR" "$@"
echo "--- DeepAEG COMPLETE ---"
echo

# ========================= 步骤 3: 运行 DeepCDR 模型 =========================
echo "--- [3/5] Running DeepCDR Model ---"
nextflow run main.nf -profile deepcdr --output_dir "$OUTPUT_DIR" "$@"
echo "--- DeepCDR COMPLETE ---"
echo

# ========================= 步骤 4: 运行 DIPK & GraphDRP 模型组 =========================
echo "--- [4/5] Running DIPK & GraphDRP Models ---"
nextflow run main.nf -profile dipk_graphdrp --output_dir "$OUTPUT_DIR" "$@"
echo "--- DIPK & GraphDRP COMPLETE ---"
echo

# ========================= 步骤 5: 运行集成投票器 =========================
echo "--- [5/5] All models complete. Running the Ensemble Voter ---"
# 注意：voter流程不需要数据文件参数，所以我们不传递 "$@"
nextflow run main.nf -profile voter --output_dir "$OUTPUT_DIR"
echo "--- Ensemble Voter COMPLETE ---"
echo

# ========================= 工作流结束 =========================
echo "==================================================="
echo "  Full Ensemble Workflow Finished Successfully!"
echo "  Final results are located in: $OUTPUT_DIR"
echo "==================================================="