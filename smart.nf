nextflow.enable.dsl=2

params.drug_file = ""
params.cell_file = ""
params.output_dir = "smart_workspace"

process SMART_PLANNER {
    publishDir "${params.output_dir}/plan", mode: 'copy'
    
    input:
    path drug_file
    path cell_file

    output:
    path "chunk_*.csv", emit: files
    path "config.sh"
    path "resource_report.txt"

    script:
    """
    #!/usr/bin/env python3
    import pandas as pd
    import subprocess
    import math
    import os

    def get_ram_mb():
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        return int(line.split()[1]) / 1024
        except: return 16384 

    def get_gpu_count():
        try:
            res = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT).decode()
            count = len([line for line in res.strip().split('\\n') if line.strip()])
            return count
        except: 
            return 0 

    total_ram = get_ram_mb()
    gpu_count = get_gpu_count()
    drugs = pd.read_csv("${drug_file}")
    cells = pd.read_csv("${cell_file}")
    n_drugs = len(drugs)
    n_cells = len(cells)

    if gpu_count == 0:
        mode = "CPU_ONLY"
        max_parallel = 1
        max_safe_pairs = 2500 
        base_load = 1024 
    else:
        mode = "GPU_ACCELERATED"
        base_load = min(20480, total_ram * 0.6)

        usable_ram = total_ram * 0.90
        incremental_margin = usable_ram - base_load
        EFFICIENCY = 250000 / (32768 * 0.9 - 20480) 
        max_safe_pairs = math.floor(incremental_margin * EFFICIENCY)
        
        if max_safe_pairs < 5000: max_safe_pairs = 5000

        if gpu_count <= 1:
            max_parallel = 1
        else:
            ram_slots = math.floor((total_ram * 0.9) / base_load)
            max_parallel = min(ram_slots, gpu_count)

    if max_parallel < 1: max_parallel = 1

    pairs_per_slot = max_safe_pairs / max_parallel
    
    chunk_size = math.floor(pairs_per_slot / n_cells)

    if chunk_size < 1: chunk_size = 1
    if chunk_size > n_drugs: chunk_size = n_drugs

    for i, start in enumerate(range(0, n_drugs, chunk_size)):
        drugs.iloc[start : start + chunk_size].to_csv(f"chunk_{i:03d}.csv", index=False)

    with open("config.sh", "w") as f:
        f.write(f"MODE={mode}\\n")
        f.write(f"MAX_PARALLEL={max_parallel}\\n")
        f.write(f"GPU_COUNT={gpu_count}\\n")
        f.write(f"TOTAL_CHUNKS={math.ceil(n_drugs/chunk_size)}\\n")

    with open("resource_report.txt", "w") as f:
        f.write(f"Mode: {mode}\\n")
        f.write(f"Hardware: RAM {total_ram:.0f}MB, GPU x{gpu_count}\\n")
        f.write(f"Model: BaseLoad {base_load:.0f}MB, MaxSafePairs {max_safe_pairs}\\n")
        f.write(f"Strategy: Parallel {max_parallel}, ChunkSize {chunk_size} drugs per file\\n")
        f.write(f"Total: {n_drugs} drugs, {n_cells} cells -> {math.ceil(n_drugs/chunk_size)} chunks")
    """
}

workflow {
    // check
    if (!params.drug_file || !params.cell_file) {
        error "Usage: nextflow run smart_planner.nf --drug_file drugs.csv --cell_file cells.csv"
    }
    SMART_PLANNER(file(params.drug_file), file(params.cell_file))
}
