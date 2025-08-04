import csv
import numpy as np
import math
import os  # 确保导入 os

# 假设您的数据文件夹路径是固定的
FOLDER_PATH = "mydata/"


def transform_ic50_value(original_ic50_value_str):
    """
    将单个原始IC50字符串值转换为变换后的浮点数值。
    变换公式: y = 1 / (1 + exp(-0.1 * x))
    """
    try:
        original_ic50 = float(original_ic50_value_str)
        # 原始代码中的变换公式
        transformed_value = 1 / (1 + pow(math.exp(original_ic50), -0.1))
        return transformed_value
    except ValueError:
        # print(f"警告: 无法将 '{original_ic50_value_str}' 转换为浮点数。")
        return None


def calculate_g_prime_scaling_factor():
    """
    计算逆变换函数 g(y) 在 y 的均值 (y_bar) 处的导数值 |g'(y_bar)|。
    这个值将作为从变换后RMSE到近似原始RMSE的缩放因子。
    g'(y) = 10 / (y * (1-y))
    """
    original_ic50_column_index = 2  # 根据您的代码，IC50在第3列 (索引2)
    transformed_y_values_list = []

    ic50_file_path = os.path.join(FOLDER_PATH, "ic50.csv")

    if not os.path.exists(ic50_file_path):
        print(f"错误: 原始数据文件 '{ic50_file_path}' 未找到。请确保路径正确。")
        return None

    try:
        with open(ic50_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头行
            for i, row in enumerate(reader):
                if len(row) > original_ic50_column_index:
                    transformed_y = transform_ic50_value(row[original_ic50_column_index])
                    if transformed_y is not None:
                        transformed_y_values_list.append(transformed_y)
                # else:
                # print(f"警告: ic50.csv 文件中第 {i+2} 行数据列数不足。")
    except Exception as e:
        print(f"读取或处理 '{ic50_file_path}' 文件时发生错误: {e}")
        return None

    if not transformed_y_values_list:
        print("错误: 未能从 'ic50.csv' 中成功处理任何有效的IC50值来计算y_bar。")
        return None

    # 使用numpy数组进行计算，并进行clipping以增加数值稳定性
    epsilon = 1e-9  # 防止 y_bar 恰好为0或1
    y_values_np = np.array(transformed_y_values_list)
    y_values_clipped = np.clip(y_values_np, epsilon, 1 - epsilon)

    y_bar = np.mean(y_values_clipped)

    # print(f"信息: 从 'ic50.csv' 计算得到的变换后IC50值的均值 (y_bar): {y_bar:.6f}")

    if y_bar <= 0 or y_bar >= 1:  # 理论上不应发生，但作为安全检查
        print(f"错误: 计算得到的 y_bar ({y_bar:.6f}) 不在 (0,1) 的有效区间内。")
        return None

    # 计算导数 g'(y_bar)
    g_prime_at_y_bar = 10 / (y_bar * (1 - y_bar))
    # print(f"信息: 在 y_bar = {y_bar:.6f} 处，逆变换的导数 g'(y_bar) = {g_prime_at_y_bar:.6f}")

    return abs(g_prime_at_y_bar)


def get_approximated_original_rmse(scaled_rmse_value):
    """
    接收一个在变换后尺度上计算的RMSE值，
    并估算其在原始IC50尺度上的RMSE。

    Args:
        scaled_rmse_value (float): 您训练得到的、在变换后尺度上的RMSE。

    Returns:
        float: 近似在原始IC50尺度上的RMSE，如果计算失败则返回None。
    """
    print("正在计算用于估算原始RMSE的缩放因子...")
    scaling_factor = calculate_g_prime_scaling_factor()

    if scaling_factor is None:
        print("错误: 无法计算缩放因子，因此无法估算原始RMSE。")
        return None

    print(f"信息: 计算得到的缩放因子 |g'(y_bar)| 为: {scaling_factor:.4f}")

    approximated_original_rmse = scaling_factor * scaled_rmse_value

    print(f"\n提供的变换后RMSE: {scaled_rmse_value:.6f}")
    print(f"据此估算的原始IC50尺度RMSE约为: {approximated_original_rmse:.6f}")

    return approximated_original_rmse

# 假设这是您从训练中得到的变换后的RMSE
transformed_rmse = 0.033481758






approximated_rmse_on_original_scale = get_approximated_original_rmse(transformed_rmse)

if approximated_rmse_on_original_scale is not None:
    print(f"\n--- 总结 ---")
    print(f"对于变换后的RMSE = {transformed_rmse},")
    print(f"估算得到的原始IC50尺度RMSE 大约为: {approximated_rmse_on_original_scale:.4f}")
else:
    print("\n--- 总结 ---")
    print("无法完成原始RMSE的估算。")