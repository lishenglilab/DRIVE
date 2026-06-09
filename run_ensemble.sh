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


cleanup_nextflow() {
    echo "INFO: Performing a smart cleanup (preserving conda cache)..."
    if [ -d "work" ]; then
        
        find work -maxdepth 1 ! -name 'conda' ! -name 'work' -exec rm -rf {} +
    fi
    rm -rf .nextflow .nextflow.log*
    echo "INFO: Cleanup complete."
}

export TERM=dumb
echo "==================================================="
echo "  Starting Full Ensemble Prediction Workflow"
echo "==================================================="


cleanup_nextflow
rm -rf results

echo "--- [1/3] Running DIPK & GraphDRP Models ---"
nextflow run main.nf -profile dipk_graphdrp "$@"
cleanup_nextflow
echo "--- [2/3] Running Part 1 Models ---"
nextflow run main.nf -profile part1 "$@"
echo "--- [3/3] Running Ensemble Analysis ---"
nextflow run main.nf -profile voter

echo "==================================================="
echo "  Full Ensemble Workflow Finished Successfully!"
echo "==================================================="
