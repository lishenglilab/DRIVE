#!/bin/bash

# ========================== 参数配置 =================================
# 请在此处填写所有文件路径和参数，确保它们正确无误

# --- 执行环境 ---
# 确保这个python指向你的conda/venv环境
PYTHON_EXECUTABLE="python"

# --- 输入文件 ---
# 假设此脚本与 test/, mydata/, GraphDRP-master/ 等目录位于同一级别
DRUG_FILE="./test/predict_all_np.csv"
CELL_FILE="./test/mu_1.csv"

# --- 数据和模型目录 ---
# 包含 PANCANCER_Genetic_feature.csv 的目录
DATA_DIR="./mydata/" 
# 包含所有.model文件的目录
MODELS_DIR="./"

# --- 输出配置 ---
# 存放所有预测结果的目录
RESULTS_DIR="./results"

# --- 性能参数 ---
# DataLoader的批处理大小
BATCH_SIZE=256
# 每次从药物文件中读取的行数
CHUNK_SIZE=10000
# DataLoader使用的工作进程数。推荐设置为CPU核心数的一半左右。
NUM_WORKERS=0
# 使用的GPU ID
GPU_ID=0

# --- 模型定义 ---
# 在这里列出所有需要运行的模型类型和对应的模型文件名
# 格式: ["模型类型"]="模型文件名"
declare -A MODELS
MODELS=(
    ["GCNNet"]="model_GCNNet_GDSC_blind_run1.model"
    ["GINConvNet"]="model_GINConvNet_GDSC_blind_run1.model"
    ["GATNet"]="model_GATNet_GDSC_blind_run1.model"
    ["GAT_GCN"]="model_GAT_GCN_GDSC_blind_run1.model"
)

# ====================================================================
#                      主   脚   本   开   始
# ====================================================================
START_TIME=$SECONDS
echo "============================================================"
echo "    GraphDRP 稳健型分块预测管道"
echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始执行..."
echo ""

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 循环遍历所有定义的模型
for model_type in "${!MODELS[@]}"
do
    model_filename=${MODELS[$model_type]}
    model_path="${MODELS_DIR}/${model_filename}"
    
    # 为每个模型创建一个专属的子目录来存放分块结果
    output_model_dir="${RESULTS_DIR}/${model_type}"
    mkdir -p "$output_model_dir"
    # 定义输出文件的基础名
    output_base_name="${output_model_dir}/predictions_${model_type}"
    
    echo "------------------------------------------------------------"
    echo ">>>>> 开始处理模型: ${model_type} <<<<<"
    echo "      - 模型文件: ${model_path}"
    echo "      - 结果将保存到: ${output_model_dir}/"
    echo "------------------------------------------------------------"

    # 检查模型文件是否存在
    if [ ! -f "$model_path" ]; then
        echo "!!!!!! 警告: 模型文件不存在 '${model_path}'。跳过此模型。!!!!!!"
        continue
    fi

    # 调用Python脚本进行分块预测
    $PYTHON_EXECUTABLE ./predict_batch.py \
        --model_path "$model_path" \
        --model_type "$model_type" \
        --drug_file "$DRUG_FILE" \
        --cell_file "$CELL_FILE" \
        --output_file "$output_base_name" \
        --data_dir "$DATA_DIR" \
        --cuda_name "cuda:${GPU_ID}" \
        --batch_size $BATCH_SIZE \
        --chunk_size $CHUNK_SIZE \
        --num_workers $NUM_WORKERS

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
