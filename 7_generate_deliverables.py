# 7_generate_deliverables.py
# 权威最终版: 根据最终审核结果，一键生成所有最终交付物——主码表Excel和回填数据Excel。

import pandas as pd
import os
import re
from tqdm import tqdm

# --- 【【【每次运行时，请根据您的项目情况修改此处的配置】】】 ---

# 1. 您本期的原始宽数据文件名
ORIGINAL_WIDE_FILE = "01_source_data/current_phase_raw_data.csv" 

# 2. 为本期数据生成的UUID地图文件名 (由 1a_prepare_new_data.py 生成)
UUID_MAP_FILE = '02_preprocessed_data/current_phase_uuid_map.csv' 

# 3. 经过您最终审核、修正后的编码结果文件名 (聚合格式)
#    这个文件由 6_convert_and_merge_results.py 生成，或由 5_review_tool.py 修正后另存
CORRECTED_DATA_FILE = "07_final_reports/Final_Coded_Report_CORRECTED_AGGREGATED.csv" 

# 4. 您上一期的原始基础码表
BASE_CODEBOOK_FILE = "01_source_data/last_phase_codebook.csv"

# 5. 您为新题目手动定稿的码表 (如果本次项目没有新题，此文件可以不存在)
NEW_QUESTION_CODEBOOK_FILE = "new_question_codebook.csv" 

# --- 输出文件配置 ---
OUTPUT_REPORTS_FOLDER = "07_final_reports"
if not os.path.exists(OUTPUT_REPORTS_FOLDER):
    os.makedirs(OUTPUT_REPORTS_FOLDER)
    
# 6. 最终码表Excel文件名
OUTPUT_CODEBOOK_EXCEL = os.path.join(OUTPUT_REPORTS_FOLDER, "Master_Codebook_Final.xlsx")
# 7. 最终回填数据Excel文件名
OUTPUT_FINAL_DATA_EXCEL = os.path.join(OUTPUT_REPORTS_FOLDER, "Final_Data_With_Codes.xlsx")
# -----------------------------------------------------------

# --- 其他配置 ---
# ID列在原始文件中的位置（0代表第一列），必须与 1a_prepare_new_data.py 中的配置一致
ID_COLUMN_INDEXES = [0]


def generate_all_deliverables():
    """
    主函数，执行生成码表和回填数据两大任务。
    """
    print("--- 开始生成所有最终交付物 ---")
    
    try:
        original_df = pd.read_csv(ORIGINAL_WIDE_FILE, dtype=str).fillna('')
        corrected_df = pd.read_csv(CORRECTED_DATA_FILE, dtype=str).fillna('')
        base_codebook_df = pd.read_csv(BASE_CODEBOOK_FILE, dtype=str)
        uuid_map_df = pd.read_csv(UUID_MAP_FILE, dtype=str).fillna('')
    except FileNotFoundError as e:
        print(f"错误：找不到必需的文件 {e.filename}。请检查顶部的配置是否正确。")
        return

    # ==============================================================================
    # --- 任务一：生成与本次编码完全匹配的最终主码表 ---
    # ==============================================================================
    print("\n--- 任务 1/2: 正在生成项目专属码表 ---")
    
    # 1. 从修正后的结果中，提取所有实际使用过的唯一code列表
    if 'code_agg' not in corrected_df.columns:
        print(f"错误：修正后的报告 '{CORRECTED_DATA_FILE}' 中缺少 'code_agg' 列。")
        return
        
    unique_codes_used = set()
    for codes in corrected_df['code_agg'].dropna():
        unique_codes_used.update(str(codes).split('; '))
    unique_codes_used.discard('') # 移除可能存在的空字符串
    
    print(f"在本次编码中共计使用了 {len(unique_codes_used)} 个唯一编码。")

    # 2. 构件一个包含所有可用编码的“超级码表”
    try:
        new_question_codebook_df = pd.read_csv(NEW_QUESTION_CODEBOOK_FILE, dtype=str)
    except FileNotFoundError:
        print(f"提示：未找到新题目码表 {NEW_QUESTION_CODEBOOK_FILE}。")
        new_question_codebook_df = pd.DataFrame()
        
    super_codebook_df = pd.concat([base_codebook_df, new_question_codebook_df], ignore_index=True).drop_duplicates(subset=['code'])

    # 3. 按需筛选，生成最终的项目专属码表
    final_codebook_df = super_codebook_df[super_codebook_df['code'].isin(unique_codes_used)]
    
    # 4. 将最终码表拆分为“旧题”和“新题”两个部分
    old_question_codes_mask = final_codebook_df['code'].isin(base_codebook_df['code'])
    final_old_question_codebook = final_codebook_df[old_question_codes_mask]
    final_new_question_codebook = final_codebook_df[~old_question_codes_mask]

    print(f"将生成包含 {len(final_old_question_codebook)} 条旧题编码和 {len(final_new_question_codebook)} 条新题编码的码表。")

    # 5. 写入Excel文件
    print("正在写入最终的码表Excel文件...")
    with pd.ExcelWriter(OUTPUT_CODEBOOK_EXCEL, engine='openpyxl') as writer:
        # 确保列的顺序
        old_cols = ['sentiment', 'net', 'subnet', 'code', 'label']
        final_old_question_codebook = final_old_question_codebook[[c for c in old_cols if c in final_old_question_codebook.columns]]
        final_old_question_codebook.sort_values(by='code').to_excel(writer, sheet_name='旧题使用码表', index=False)
        
        if not final_new_question_codebook.empty:
            new_cols = ['net', 'subnet', 'code', 'label'] # 新题码表不含sentiment
            final_new_question_codebook = final_new_question_codebook[[c for c in new_cols if c in final_new_question_codebook.columns]]
            final_new_question_codebook.sort_values(by='code').to_excel(writer, sheet_name='新题使用码表', index=False)
    print(f"成功生成最终码表文件: {OUTPUT_CODEBOOK_EXCEL}")

    # ==============================================================================
    # --- 任务二：将编码结果回填到原始表格 ---
    # ==============================================================================
    print("\n--- 任务 2/2: 正在将编码结果回填到原始表格 ---")

    id_col_names = original_df.columns[ID_COLUMN_INDEXES].tolist()
    if 'uuid' not in corrected_df.columns:
        print(f"错误：审核后的文件 '{CORRECTED_DATA_FILE}' 中缺少'uuid'列，无法进行回填。")
        return
        
    merged_df = pd.merge(corrected_df[['uuid', 'code_agg']], uuid_map_df, on='uuid', how='left')
    merged_df.dropna(subset=['original_row_id', 'original_question_col'], inplace=True)

    codes_df = pd.DataFrame(index=original_df.index, columns=original_df.columns).fillna('')
    original_df['__temp_id__'] = original_df[id_col_names].astype(str).agg('_'.join, axis=1)

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Reshaping data"):
        original_id_val = row['original_row_id']
        original_col_name = row['original_question_col']
        code_value = row['code_agg']
        target_row_indices = original_df.index[original_df['__temp_id__'] == original_id_val].tolist()
        if target_row_indices:
            codes_df.loc[target_row_indices[0], original_col_name] = code_value
    
    final_df = original_df.copy()
    question_cols = [col for col in original_df.columns if col not in id_col_names and col != '__temp_id__']
    
    for q_col in question_cols:
        code_col_name = f"{q_col}_code"
        if q_col in codes_df.columns:
            final_df[code_col_name] = codes_df[q_col]
        else:
            final_df[code_col_name] = ''
        
    final_df.drop(columns=['__temp_id__'], inplace=True)
    
    new_col_order = []
    for col in original_df.columns:
        if col == '__temp_id__': continue
        new_col_order.append(col)
        code_col_name = f"{col}_code"
        if code_col_name in final_df.columns:
            new_col_order.append(code_col_name)
    
    final_df = final_df[[col for col in new_col_order if col in final_df.columns]]
    
    final_df.to_excel(OUTPUT_FINAL_DATA_EXCEL, index=False)
    print(f"成功生成回填数据文件: {OUTPUT_FINAL_DATA_EXCEL}")

if __name__ == "__main__":
    from tqdm import tqdm
    try:
        import openpyxl
    except ImportError:
        print("错误：缺少 openpyxl 库。请运行: pip install openpyxl")
        exit()
    generate_all_deliverables()