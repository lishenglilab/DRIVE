#!/bin/bash
#
# ==============================================================================
#   DRP Ensemble Workflow - Full Sequential Run (Optimized Order)
# ==============================================================================
#
#   This script sequentially executes all prediction workflows in an optimized order
#   to minimize environment setup time.
#   1. Runs DIPK/GraphDRP in a clean state.
#   2. Cleans up again.
#   3. Runs Part1 and the Voter sequentially, allowing them to share the same
#      Conda environment for better performance.
#
# ==============================================================================

# --- Script settings ---
set -euo pipefail

# --- Conda Initialization ---
# 确保 Conda 在脚本环境中被正确初始化
CONDA_BASE_PATH=$(conda info --base)
if [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
    echo "INFO: Successfully initialized Conda for this script's environment."
else
    echo "ERROR: Could not find conda.sh to initialize the Conda environment." >&2
    exit 1
fi


# --- Cleanup Function Definition ---
# 定义一个强力清理函数，删除所有 Nextflow 状态
cleanup_nextflow() {
    echo "INFO: Performing a full cleanup of Nextflow cache and work directories..."
    # 使用 -f 选项确保即使目录不存在也不会报错
    rm -rf work .nextflow .nextflow.log*
    # 运行 nextflow clean 作为额外的保险措施
    nextflow clean -f -k || true
    echo "INFO: Cleanup complete."
}

echo "==================================================="
echo "  Starting Full Ensemble Prediction Workflow"
echo "==================================================="
echo "INFO: All results will be published to the default Nextflow output directory ('results/')."
echo

# --- 初始工作区清理 ---
cleanup_nextflow
# 仅在工作流最开始时清理结果目录
rm -rf results

# ========================= Step 1: Run DIPK & GraphDRP Models =========================
echo "--- [1/3] Running DIPK & GraphDRP Models (Independent Environment) ---"
nextflow run main.nf -profile dipk_graphdrp "$@"
echo "--- DIPK & GraphDRP Models COMPLETE ---"
echo

# --- 中间状态清理 ---
# 【【【 核心修复：使用强力清理函数替换弱清理命令 】】】
# 这将确保第二步在一个与第一步完全相同的、绝对干净的状态下开始。
cleanup_nextflow

# ========================= Step 2: Run Part 1 Models =========================
echo "--- [2/3] Running Part 1 Models ---"
nextflow run main.nf -profile part1 "$@"
echo "--- Part 1 Models COMPLETE ---"
echo

# ========================= Step 3: Run Ensemble Analysis =========================
echo "--- [3/3] Running the Ensemble Analysis (Reusing Part 1 Environment) ---"
# 这里不进行清理。这允许 voter 进程复用 Part 1 运行时缓存的环境，
# 从而显著提高执行速度。
nextflow run main.nf -profile voter
echo "--- Ensemble Analysis COMPLETE ---"
echo

# ========================= Workflow End =========================
echo "==================================================="
echo "  Full Ensemble Workflow Finished Successfully!"
echo "  Final results are located in the 'results/' directory."
echo "==================================================="