# 1a_prepare_new_data.py
# 权威最终版: 专用于处理新的未编码数据，并已恢复“去后缀”功能。

import pandas as pd
import uuid
import os
import re
from tqdm import tqdm

# --- 用户配置区 ---
# 您的原始文件名
INPUT_WIDE_FILE = '01_source_data/current_phase_raw_data.csv' 
# 您希望输出的标准长格式文件名
OUTPUT_LONG_FILE = '02_preprocessed_data/current_phase_tidy.csv'
# 为本期数据生成的UUID地图文件名
UUID_MAP_FILE = '02_preprocessed_data/current_phase_uuid_map.csv'
# ID列在原始文件中的位置（0代表第一列）
ID_COLUMN_INDEXES = [0] 

def prepare_uncoded_data(input_file, output_file_long, output_file_map):
    """
    读取宽格式的未编码文件，为其每一个非空回答单元格生成一个UUID，
    并输出标准长格式文件和UUID地图文件。
    """
    print(f"--- 正在处理新的未编码文件: {input_file} ---")
    
    output_dir = os.path.dirname(output_file_long)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建文件夹: {output_dir}")

    try:
        df_wide = pd.read_csv(input_file, dtype=str).fillna('')
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}。请检查文件名和路径。")
        return

    if df_wide.empty:
        print("错误：输入文件为空。")
        return

    try:
        id_col_names = df_wide.columns[ID_COLUMN_INDEXES].tolist()
    except IndexError:
        print(f"错误：配置的ID列位置 {ID_COLUMN_INDEXES} 超出文件范围。")
        return
            
    print(f"已将列 {id_col_names} 识别为ID列。")
    
    df_wide['__temp_id__'] = df_wide[id_col_names].astype(str).agg('_'.join, axis=1)
    
    # 所有非ID列都是问题回答列
    data_columns = [col for col in df_wide.columns if col not in id_col_names and col != '__temp_id__']
    
    all_long_data = []
    
    # 1. 宽表转长表
    for index, row in tqdm(df_wide.iterrows(), total=len(df_wide), desc="1/3: Reshaping Data"):
        row_id_val = row['__temp_id__']
        for q_col in data_columns:
            answer_text = row[q_col]
            if answer_text:
                record = {
                    'original_row_id': row_id_val,
                    'question': q_col, # 此时的q_col可能包含.1, .2等后缀
                    'text': answer_text
                }
                for id_name in id_col_names:
                    record[id_name] = row[id_name]
                all_long_data.append(record)

    if not all_long_data:
        print("警告：处理后未发现任何有效的回答记录。")
        return
        
    df_long = pd.DataFrame(all_long_data)
    
    # 2. 【【【核心修正：恢复“去后缀”功能】】】
    print("\n--- 2/3: 正在移除问题文本中的'.1', '.2'等后缀 ---")
    df_long['original_question_col'] = df_long['question'] # 保留带后缀的原始列名用于地图
    df_long['question'] = df_long['question'].str.replace(r'\.\d+$', '', regex=True)
    print("后缀移除完成。")

    # 3. 生成UUID和最终文件
    print("\n--- 3/3: 正在生成UUID和最终文件 ---")
    df_long['uuid'] = [str(uuid.uuid4()) for _ in range(len(df_long))]

    # 创建UUID地图
    uuid_map_df = df_long[['uuid', 'original_row_id', 'original_question_col']]
    
    # 整理并保存主文件
    final_cols = ['uuid'] + id_col_names + ['question', 'text']
    final_df = df_long[[col for col in final_cols if col in df_long.columns]]

    final_df.to_csv(output_file_long, index=False, encoding='utf-8-sig')
    uuid_map_df.to_csv(output_file_map, index=False, encoding='utf-8-sig')
    
    print("-" * 50)
    print("新数据预处理成功！")
    print(f"已生成标准长格式文件: {output_file_long} (共 {len(final_df)} 条有效记录)")
    print(f"已生成UUID地图文件: {output_file_map}")
    print("-" * 50)


if __name__ == "__main__":
    prepare_uncoded_data(INPUT_WIDE_FILE, OUTPUT_LONG_FILE, UUID_MAP_FILE)