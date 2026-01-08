#!/bin/bash
set -euo pipefail

# --- Conda Initialization ---
CONDA_BASE_PATH=$(conda info --base)
if [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
    echo "INFO: Successfully initialized Conda."
else
    echo "ERROR: Could not find conda.sh" >&2
    exit 1
fi

# --- 智能清理函数：只删缓存，不删 Conda 环境 ---
cleanup_nextflow() {
    echo "INFO: Performing a smart cleanup (preserving conda cache)..."
    if [ -d "work" ]; then
        # 寻找 work 目录下除了 'conda' 以外的所有文件夹并删除
        find work -maxdepth 1 ! -name 'conda' ! -name 'work' -exec rm -rf {} +
    fi
    rm -rf .nextflow .nextflow.log*
    echo "INFO: Cleanup complete."
}

export TERM=dumb
echo "==================================================="
echo "  Starting Full Ensemble Prediction Workflow"
echo "==================================================="

# 1. 运行前清理 (保留环境)
cleanup_nextflow
rm -rf results

# 2. 运行 DIPK & GraphDRP
echo "--- [1/3] Running DIPK & GraphDRP Models ---"
nextflow run main.nf -profile dipk_graphdrp "$@"

# 3. 中间清理 (保留环境)
cleanup_nextflow

# 4. 运行 Part 1
echo "--- [2/3] Running Part 1 Models ---"
nextflow run main.nf -profile part1 "$@"

# 5. 运行 Ensemble (不清理，直接复用 Part 1 环境)
echo "--- [3/3] Running Ensemble Analysis ---"
nextflow run main.nf -profile voter

echo "==================================================="
echo "  Full Ensemble Workflow Finished Successfully!"
echo "==================================================="
