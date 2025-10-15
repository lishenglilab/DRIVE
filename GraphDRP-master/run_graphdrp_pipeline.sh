#!/bin/bash

# ========================== 参数配置 =================================
# 请在此处填写所有文件路径和参数，确保它们正确无误

# --- 执行环境 ---
# 确保这个python指向你的conda/venv环境
PYTHON_EXECUTABLE="python"
# 【重要】确保这里的脚本名和您保存Python代码的文件名一致
PYTHON_SCRIPT="predict_graphdrp.py" 

# --- 输入文件 ---
DRUG_FILE="./CRC/drug_depmap.csv"
CELL_FILE="./CRC/mu_tcga.csv"

# --- 数据和模型目录 ---
# 包含 mut_dict.pkl 的目录
DATA_DIR="./mydata/" 
# 包含所有.model文件的目录
MODELS_DIR="./"

# --- 输出配置 ---
# 存放所有预测结果的目录
RESULTS_DIR="./results"

# --- 【【【 修改点 】】】: 新增并配置性能参数 ---
# DataLoader的批处理大小
BATCH_SIZE=256
# 每次从药物文件中读取的行数
CHUNK_SIZE=10000
# 【新增】一次性加载进行预测的细胞系数量
CELL_CHUNK_SIZE=32
# 【新增】用于SMILES转图的CPU核心数
CPU_WORKERS=8
# DataLoader使用的工作进程数
NUM_WORKERS=4
# 使用的GPU ID
GPU_ID=0

# --- 模型定义 ---
declare -A MODELS
MODELS=(
    ["GATNet"]="model_GATNet_GDSC_blind_run1.model"
    ["GAT_GCN"]="model_GAT_GCN_GDSC_blind_run1.model"
)

# ====================================================================
#                      主   脚   本   开   始
# ====================================================================
START_TIME=$SECONDS
echo "============================================================"
echo "    GraphDRP 稳健型并行预测管道 (v2.2 - 完全配置版)"
echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始执行..."
echo ""

mkdir -p "$RESULTS_DIR"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "!!!!!! 严重错误: 预测脚本 '$PYTHON_SCRIPT' 不存在。请检查文件名。!!!!!!"
    exit 1
fi

for model_type in "${!MODELS[@]}"
do
    model_filename=${MODELS[$model_type]}
    model_path="${MODELS_DIR}/${model_filename}"
    
    output_model_dir="${RESULTS_DIR}/${model_type}"
    mkdir -p "$output_model_dir"
    
    echo "------------------------------------------------------------"
    echo ">>>>> 开始处理模型: ${model_type} <<<<<"
    echo "      - 模型文件: ${model_path}"
    echo "      - 结果将保存到: ${output_model_dir}/"
    echo "------------------------------------------------------------"

    if [ ! -f "$model_path" ]; then
        echo "!!!!!! 警告: 模型文件不存在 '${model_path}'。跳过此模型。!!!!!!"
        continue
    fi

    # 【【【 修改点 】】】: 调用Python脚本，传入所有参数，包括新增的并行化参数
    $PYTHON_EXECUTABLE $PYTHON_SCRIPT \
        --model_path "$model_path" \
        --model_type "$model_type" \
        --drug_file "$DRUG_FILE" \
        --cell_file "$CELL_FILE" \
        --output_dir "$output_model_dir" \
        --output_prefix "predictions" \
        --data_dir "$DATA_DIR" \
        --cuda_name "cuda:${GPU_ID}" \
        --batch_size $BATCH_SIZE \
        --chunk_size $CHUNK_SIZE \
        --num_workers $NUM_WORKERS \
        --cell_chunk_size $CELL_CHUNK_SIZE \
        --cpu_workers_smiles $CPU_WORKERS

    if [ $? -ne 0 ]; then
        echo "!!!!!! 警告: 模型 '${model_type}' 的预测流程中发生错误。!!!!!!"
    else
        echo "      ✅ 模型 ${model_type} 所有块处理完毕。"
    fi
    echo ""
done

ELAPSED_TIME=$(( SECONDS - START_TIME ))
echo "============================================================"
echo "                      所有任务完成"
echo "============================================================"
echo "总耗时: ${ELAPSED_TIME} 秒"
echo "所有模型的预测结果已按模型分类保存在目录: $RESULTS_DIR"
echo "============================================================"