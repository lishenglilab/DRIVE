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

# This function performs a complete cleanup of the Nextflow state.
cleanup_nextflow() {
    echo "INFO: Performing a full cleanup of Nextflow cache and work directories..."
    nextflow clean -f || true
    rm -rf work .nextflow .nextflow.log*
    echo "INFO: Cleanup complete."
}

echo "==================================================="
echo "  Starting Full Ensemble Prediction Workflow"
echo "==================================================="
echo "INFO: All results will be published to the default Nextflow output directory ('results/')."
echo

# --- Initial Cleanup ---
cleanup_nextflow
# Clean the results directory only at the very beginning.
rm -rf results

# ========================= Step 1: Run DIPK & GraphDRP Models =========================
echo "--- [1/3] Running DIPK & GraphDRP Models (Independent Environment) ---"
nextflow run main.nf -profile dipk_graphdrp "$@"
echo "--- DIPK & GraphDRP Models COMPLETE ---"
echo

# --- Intermediate Cleanup ---
# Clean up before starting the next group to ensure isolation.
nextflow clean -f

# ========================= Step 2: Run Part 1 Models =========================
echo "--- [2/3] Running Part 1 Models ---"
nextflow run main.nf -profile part1 "$@"
echo "--- Part 1 Models COMPLETE ---"
echo

# ========================= Step 3: Run Ensemble Analysis =========================
echo "--- [3/3] Running the Ensemble Analysis (Reusing Part 1 Environment) ---"
# NO cleanup is performed here. This allows the voter to reuse the cached
# environment from the Part 1 run, which is significantly faster.
nextflow run main.nf -profile voter
echo "--- Ensemble Analysis COMPLETE ---"
echo

# ========================= Workflow End =========================
echo "==================================================="
echo "  Full Ensemble Workflow Finished Successfully!"
echo "  Final results are located in the 'results/' directory."
echo "==================================================="