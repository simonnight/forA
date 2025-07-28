# utils/file_utils.py
import os
import glob
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def load_codebook(path):
    """加载码表文件"""
    try:
        df = pd.read_csv(path)
        # 确保所有列都存在，即使为空
        expected_columns = ['code', 'label', 'net', 'subnet', 'sentiment', 'question']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
        return df
    except Exception as e:
        print(f"加载码表失败: {e}")
        # 返回一个空的DataFrame，但包含预期的列
        return pd.DataFrame(columns=['code', 'label', 'net', 'subnet', 'sentiment', 'question'])

def list_raw_batches(folder):
    """递归列出所有原始批次文件"""
    if not os.path.exists(folder):
        print(f"警告: 原始数据目录 {folder} 不存在")
        return []
    pattern = os.path.join(folder, "**", "coded_batch_*.jsonl")
    return sorted(glob.glob(pattern, recursive=True))

def list_corrected_batches(folder):
    """递归列出所有已修正批次文件 (虽然通常修正文件在顶层目录)"""
    if not os.path.exists(folder):
        print(f"警告: 修正结果目录 {folder} 不存在")
        return []
    pattern = os.path.join(folder, "**", "*_CORRECTED.jsonl")
    return sorted(glob.glob(pattern, recursive=True))

def get_batch_name(file_path):
    """从文件路径获取批次名称（相对于 05_coded_results 或 06_reviewed_results 的路径）"""
    # 移除文件扩展名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # 移除可能的 _CORRECTED 后缀
    if base_name.endswith("_CORRECTED"):
        base_name = base_name[:-len("_CORRECTED")]
    return base_name

def get_relative_path(file_path, base_folder):
    """获取文件相对于基础文件夹的路径"""
    try:
        # os.path.relpath 返回 file_path 相对于 base_folder 的路径
        rel_path = os.path.relpath(file_path, base_folder)
        # 移除 .jsonl 扩展名
        return os.path.splitext(rel_path)[0]
    except ValueError:
        # 如果 file_path 不在 base_folder 下，返回文件名
        return os.path.splitext(os.path.basename(file_path))[0]

def get_corrected_path(raw_file_path, corrected_folder="06_reviewed_results", raw_base_folder="05_coded_results"):
    """根据原始文件路径（包含子目录）生成修正文件路径"""
    # 获取相对于原始数据目录的路径（不含扩展名）
    rel_path_without_ext = get_relative_path(raw_file_path, raw_base_folder)
    # 构造修正文件的完整路径，保持子目录结构
    corrected_filename = f"{rel_path_without_ext}_CORRECTED.jsonl"
    return os.path.join(corrected_folder, corrected_filename)

# --- 新增辅助函数 ---
def get_raw_path_from_corrected(corrected_file_path, raw_folder="05_coded_results", corrected_base_folder="06_reviewed_results"):
    """根据修正文件路径反推原始文件路径"""
    try:
        rel_path = os.path.relpath(corrected_file_path, corrected_base_folder)
        if rel_path.endswith("_CORRECTED.jsonl"):
            raw_rel_path = rel_path[:-len("_CORRECTED.jsonl")] + ".jsonl"
            return os.path.join(raw_folder, raw_rel_path)
    except ValueError:
        pass
    return None