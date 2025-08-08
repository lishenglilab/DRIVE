#!/bin/bash

# ==============================================================================
#  GADRP 批量预测控制器脚本 (run_all_chunks.sh)
# ==============================================================================
#
#  功能:
#  - 自动按顺序处理从 Chunk 1 到 Chunk 21 的所有预测任务。
#  - 每个 Chunk 都在一个独立的、干净的 Python 进程中运行，彻底解决内存累积问题。
#  - 如果任何一个 Chunk 的预测失败，脚本会报错并停止，方便定位问题。
#
#  使用方法:
#  1. 确认下面的 PYTHON_EXECUTABLE 和参数配置正确。
#  2. 在终端中运行 `chmod +x run_all_chunks.sh` 来赋予脚本执行权限。
#  3. 运行 `./run_all_chunks.sh` 来启动全部任务。
#
# ==============================================================================

# --- 配置区 ---

# 1. 设置你的Python解释器路径
#    (如果你的conda环境已经激活，直接用 "python" 即可)
PYTHON_EXECUTABLE="python"

# 2. 设置要处理的Chunk总数
TOTAL_CHUNKS=21


# --- 脚本主体 (无需修改) ---

echo "======================================================="
echo "  启动 GADRP 批量预测任务"
echo "  将要处理总共 $TOTAL_CHUNKS 个大块文件。"
echo "======================================================="
echo ""

# 使用 for 循环，从 1 迭代到 TOTAL_CHUNKS
for i in $(seq 21 $TOTAL_CHUNKS)
do
    echo "-------------------------------------------------------"
    echo "  正在启动对 Chunk $i/$TOTAL_CHUNKS 的预测任务..."
    echo "-------------------------------------------------------"
    
    # 调用Python脚本，传递所有必要的参数。
    # 核心是 --chunk_num $i，它告诉Python脚本这次要处理哪个块。
    # 【【【重要】】】请再次确认下面的所有文件路径和参数都是正确的！
    
    $PYTHON_EXECUTABLE predict_all_debugg.py \
        --chunk_num $i \
        --new_drug_feature_file './test/drug_with_conditions.csv' \
        --new_exp_file './test/exp_1.csv' \
        --new_cn_file './test/cnv_1.csv' \
        --new_meth_file './test/mu_1.csv' \
        --new_mirna_file './test/mi_1.csv' \
        --temp_chunk_dir './drug_temp_chunks' \
        --big_chunk_size 100000 \
        --small_chunk_size 1000 \
        --model_path './model/saved_models/best_model_drug_blind_fold2.pth' \
        --output_file './predictions/GADRP_predictions.csv' \
        --ae_epochs 2500 \
        --cell_index_file './mydata/cell_line/cell_index.csv' \
        --train_exp_file './mydata/cell_line/exp_process.csv' \
        --train_cn_file './mydata/cell_line/cn_process.csv' \
        --train_meth_file './mydata/cell_line/meth_process.csv' \
        --train_mirna_file './mydata/cell_line/mi_process.csv' \
        --drug_physicochemical_file './mydata/drug/269dim.csv' \
        --all_drugs_feature_file './mydata/drug/drug_with_conditions.csv' \
        --drug_response_file './mydata/pair/drug_response.csv'

    # 检查上一个命令（Python脚本）的退出状态码。
    # 0 代表成功，任何非0值代表失败。
    if [ $? -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "  严重错误: Chunk $i 的预测任务失败！"
        echo "  请检查上面的日志输出以获取详细的错误信息。"
        echo "  为了防止后续错误，整个流程已停止。"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1 # 立即退出脚本
    fi

    echo ""
    echo "--- ✓ Chunk $i 处理完成。---"
    echo ""
done

echo ""
echo "======================================================="
echo "  🎉 恭喜！所有 $TOTAL_CHUNKS 个大块均已成功处理！"
echo "======================================================="
