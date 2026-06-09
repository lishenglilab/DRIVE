#!/bin/bash
set -euo pipefail

export NXF_CONDA_CACHEDIR=/opt/conda/envs
export NXF_OFFLINE=true

echo "==================================================="
echo "  DRP Unified Pipeline - Docker Mode Start"
echo "==================================================="


mkdir -p /app/results
find /app/results -mindepth 1 -delete || true
rm -rf /app/work /app/.nextflow


echo "INFO: Step 1/3 - Running DIPK & GraphDRP Models..."
nextflow run /app/main_docker.nf --entry "dipk_graphdrp" -with-conda -ansi disabled

echo "INFO: Step 2/3 - Running Part 1 Models..."
nextflow run /app/main_docker.nf --entry "part1" -with-conda -ansi disabled

echo "INFO: Step 3/3 - Final Ensemble Voting..."
nextflow run /app/main_docker.nf --entry "voter" -with-conda -ansi disabled

echo "==================================================="
echo "  Workflow Finished! Results are in /app/results"
echo "==================================================="
