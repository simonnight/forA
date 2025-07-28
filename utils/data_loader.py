# utils/data_loader.py
import json
import streamlit as st
import os

@st.cache_data(ttl=3600)
def load_jsonl(file_path):
    """加载JSONL文件"""
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return []

def append_to_jsonl(file_path, data):
    """追加数据到JSONL文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"写入文件 {file_path} 失败: {e}")
        return False

def write_jsonl(file_path, data_list):
    """写入完整的JSONL文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"写入文件 {file_path} 失败: {e}")
        return False