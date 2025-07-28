# 3_run_batch_coding.py
# 权威最终版 (多线程 + 增强版数据清洗): 集成五层智能过滤、多线程加速、批次内去重，并修正所有已知错误。

import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import google.api_core.exceptions
import os
import json
import time
import pickle
from tqdm import tqdm
import traceback
import re
import shutil
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from threading import Lock, local
import glob

# --- 【【【每次运行时，请修改此处的批处理配置】】】 ---
# 1. 精确复制/粘贴您当前要处理的问题文本
BATCH_QUESTION_TEXT = "评分的原因"
# 2. 指定对应的批次文件名 (确保文件在04_input_batches_to_code文件夹内)
BATCH_INPUT_FILE = "04_input_batches_to_code/batch_问题A.csv"
# 3. 设置并发线程数
NUM_THREADS = 10  # 根据你的API配额和系统资源调整
# ----------------------------------------------------

# --- 其他配置 ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("错误：请先设置 GOOGLE_API_KEY 环境变量。")
    exit()

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash-lite"
SIMILARITY_COPY_THRESHOLD = 0.35 # 适度放宽的阈值

# --- 文件路径定义 ---
CODEBOOK_FILE = "01_source_data/last_phase_codebook.csv"
KB_FOLDER = "03_knowledge_base"
ANSWER_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "answer.index")
QUESTION_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "question.index")
DATA_MAP_FILE = os.path.join(KB_FOLDER, "data_map.pkl")
UNIQUE_QUESTIONS_FILE = os.path.join(KB_FOLDER, "unique_questions.pkl")
QUESTION_TO_SUB_INDEX_MAP_FILE = os.path.join(KB_FOLDER, "question_to_sub_index_map.pkl")
RESULTS_FOLDER = "05_coded_results/old_questions"
REVIEWED_RESULTS_FOLDER = "06_reviewed_results"
for folder in [RESULTS_FOLDER, REVIEWED_RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 无意义内容过滤器配置 ---
MEANINGLESS_WORDS = {"ok", "good", "nice", "fine", "test", "null", "none", "na", "n/a", "好", "好的", "好好好", "好的好的", "不错", "可以", "没问题", "测试", "test", "无", "没有", "无意见", "不详", "不知道", "不清楚", "1", "11", "111", "666"}
MEANINGLESS_CODE_JSON = [{"sentiment": "负面", "net": "其他", "subnet": "无效回答", "code": "999", "label": "无明确意义回答"}]

# ==============================================================================
# --- 辅助函数定义区 ---
# ==============================================================================
def is_meaningless(text: str) -> bool:
    if not text: return True
    cleaned_text = text.strip()
    if not cleaned_text: return True
    if cleaned_text.lower() in MEANINGLESS_WORDS: return True
    if cleaned_text.isdigit(): return True
    if re.fullmatch(r"[^\w\u4e00-\u9fa5]+", cleaned_text): return True
    return False

def find_best_matching_question_index(question_index, unique_questions, new_question: str) -> int:
    response = genai.embed_content(model=EMBEDDING_MODEL, content=new_question, task_type="RETRIEVAL_QUERY")
    query_vector = np.array([response['embedding']]).astype('float32')
    distances, indices = question_index.search(query_vector, 1)
    return indices[0][0]

def format_examples_concise(df_slice, codebook):
    """【API经济优化】格式化案例，对长文本进行截断"""
    lines = []
    for _, row in df_slice.iterrows():
        code = str(row['code'])
        if code in codebook.index:
            code_info = codebook.loc[code]
            text_preview = row['text'][:80] + '...' if len(row['text']) > 80 else row['text']
            lines.append(f"- 回答: \"{text_preview}\"\n  -> 编码: {{ code: {code}, label: '{code_info.label}' }}")
    return "\n".join(lines)

def clean_code_info(code_series, code: str) -> dict:
    if isinstance(code_series, pd.DataFrame): record = code_series.iloc[0]
    else: record = code_series
    sentiment = record.sentiment if pd.notna(record.sentiment) else ""
    net = record.net if pd.notna(record.net) else ""
    subnet = record.subnet if pd.notna(record.subnet) else ""
    label = record.label if pd.notna(record.label) else ""
    return {'sentiment': str(sentiment), 'net': str(net), 'subnet': str(subnet), 'code': str(code), 'label': str(label)}

def clean_json_string(raw_text: str) -> str:
    start_brace = raw_text.find('{')
    start_bracket = raw_text.find('[')
    if start_brace == -1: start_brace = float('inf')
    if start_bracket == -1: start_bracket = float('inf')
    start_index = min(start_brace, start_bracket)
    end_brace = raw_text.rfind('}')
    end_bracket = raw_text.rfind(']')
    end_index = max(end_brace, end_bracket)
    if start_index < float('inf') and end_index != -1:
        return raw_text[start_index : end_index + 1]
    return raw_text

thread_local_storage = local()
def get_thread_local_model():
    if not hasattr(thread_local_storage, "model"):
        system_instruction = "作为质性研究编码专家，你的任务是：\n1. 识别反馈中的所有独立观点。\n2. 为每个观点匹配一个基本一致的现有编码。\n3. 若现有编码过宽泛，则在相同层级下建议更具体的新编码。\n4. 必须严格返回JSON格式。"
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        thread_local_storage.model = genai.GenerativeModel(GENERATION_MODEL, system_instruction=system_instruction, generation_config=generation_config)
    return thread_local_storage.model

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted))
def generate_content_with_retry(user_prompt):
    model = get_thread_local_model()
    return model.generate_content(user_prompt)

def estimate_k_value(text: str) -> int:
    text_length = len(text)
    if text_length <= 5:   return 3
    elif text_length <= 15: return 4
    else:                    return 7

def rag_pipeline_final(resources, new_text: str, similarity_threshold: float, preloaded_sub_index, original_indices_for_sub_index):
    try:
        df_map = resources["df_map"]
        codebook_df = resources["codebook_df"]
        answer_index = resources["answer_index"]
        
        dynamic_k = estimate_k_value(new_text)

        if preloaded_sub_index.ntotal > 0 and similarity_threshold is not None:
            response = genai.embed_content(model=EMBEDDING_MODEL, content=new_text, task_type="RETRIEVAL_QUERY")
            query_vector = np.array([response['embedding']]).astype('float32')
            distances, sub_indices = preloaded_sub_index.search(query_vector, 1)
            if distances.size > 0 and distances[0][0] < similarity_threshold:
                best_match_original_idx = original_indices_for_sub_index[sub_indices[0][0]]
                codes_to_copy = df_map[df_map.index == best_match_original_idx]['code'].tolist()
                unique_codes = sorted(list(set(codes_to_copy)))
                copied_codes = [clean_code_info(codebook_df.loc[str(c)]) for c in unique_codes if str(c) in codebook_df.index]
                return copied_codes, "SIMILARITY_MATCH_COPY"
        
        prompt_mode, relevant_examples = "zero_shot", pd.DataFrame()
        if 'query_vector' not in locals():
            response = genai.embed_content(model=EMBEDDING_MODEL, content=new_text, task_type="RETRIEVAL_QUERY")
            query_vector = np.array([response['embedding']]).astype('float32')
        if preloaded_sub_index.ntotal > 0:
            distances, sub_indices = preloaded_sub_index.search(query_vector, min(dynamic_k, preloaded_sub_index.ntotal))
            if distances.size > 0 and distances[0][0] < 0.9:
                prompt_mode, original_indices = "standard", [original_indices_for_sub_index[i] for i in sub_indices[0]]
                relevant_examples = df_map.loc[original_indices]
        if prompt_mode == "zero_shot":
            distances, indices = answer_index.search(query_vector, dynamic_k)
            if distances.size > 0 and distances[0][0] < 0.9:
                prompt_mode, relevant_examples = "cross_context_warning", df_map.iloc[indices[0]]
        
        reference_section = ""
        if prompt_mode == "zero_shot":
            reference_section = f"## 码表参考\n请参考您知识中的完整码表进行判断。"
        else:
            concise_examples = format_examples_concise(relevant_examples, codebook_df)
            unique_codes_from_examples = relevant_examples['code'].unique().astype(str)
            relevant_code_info_df = codebook_df.loc[codebook_df.index.intersection(unique_codes_from_examples)]
            concise_definitions = "\n".join([f"- {idx}: {row['label']}" for idx, row in relevant_code_info_df.iterrows()])
            
            top_example_code = str(relevant_examples.iloc[0]['code'])
            contextual_codes_str = ""
            if top_example_code in codebook_df.index:
                top_example_info = codebook_df.loc[top_example_code]
                if isinstance(top_example_info, pd.DataFrame): top_example_info = top_example_info.iloc[0]
                main_net, main_subnet = top_example_info['net'], top_example_info['subnet']
                contextual_codes_df = codebook_df[(codebook_df['net'] == main_net) & (codebook_df['subnet'] == main_subnet)]
                if not contextual_codes_df.empty:
                    concise_context_codes = "\n".join([f"- {idx}: {row['label']}" for idx, row in contextual_codes_df.iterrows()])
                    contextual_codes_str = f"## 相关上下文编码\n{concise_context_codes}"
            
            reference_section = f"## 相关历史案例 (摘要)\n{concise_examples}\n\n## 相关编码定义\n{concise_definitions}\n\n{contextual_codes_str}"
        
        user_prompt = f"""
# 任务：多维度编码
请为以下“待编码文本”进行专业的多层级编码。
---
## 待编码文本
**问题**: "{BATCH_QUESTION_TEXT}"
**回答**: "{new_text}"
---
# 参考信息
{reference_section}
---
# 分析与决策指令
请严格遵循以下“层级归因与创造”的思考过程：
**第一优先级：直接匹配现有编码**
-   分析“回答”文本中包含的所有独立观点。
-   对每一个观点，检查“参考信息”中是否存在一个`label`，其**核心意思**与该观点比较一致。
-   **如果找到这样的强匹配项（例如，回答是“服务很好”，案例中有`label`是“服务态度好/周到”），你【必须】优先直接使用该编码**，并将其`is_new_suggestion`设为`false`。

**第二优先级：智能创造新编码**
-   **仅在**某个观点【明确无法】与任何一个已有`label`观点比较一致，或者所有相关`label`都过于宽泛时，才为该观点启动创造流程。
-   **创造流程 (层级归因)**:
    a. **生成新Label**: 首先，为这个新观点提炼一个全新的、简短清晰的核心`label`。
    b. **层级归因**: 拿着这个新`label`，回到`k`个“相关历史案例”中，逐层为它寻找有没有最合适的：
        i.  **寻找Sentiment (只选不创)**: 从参考案例中，为新`label`【选择】一个最贴切的情感（只能是“正面”或“负面”）。
        ii. **寻找或创造Net**: 在上一步选定的情感分类下，从参考案例中寻找适合新`label`的`net`。如果找不到，则【创造】一个全新的`net`。
        iii. **寻找或创造Subnet**: 在已确定的`net`分类下，寻找适合新`label`的`subnet`。如果找不到，则【创造】一个简短清晰的新`subnet`。

# 输出要求
请以一个【JSON数组】的格式输出你的最终决策。
- 如果是匹配到的旧编码，请填写完整的编码信息。
- 如果是创造的新编码，请将`code`字段统一标记为`"x"`。
- 如果没有任何编码适用，请返回一个空数组 `[]`。
[
  {{
    "sentiment": "正面/负面",
    "net": "主类别",
    "subnet": "子类别",
    "code": "一个有效的数字编码或'x'",
    "label": "具体标签"
  }}
]
"""
        
        chat_response = generate_content_with_retry(user_prompt)
        cleaned_text = clean_json_string(chat_response.text)
        if not cleaned_text:
            return [{"error": "API returned empty response after cleaning."}], "ERROR"
        return json.loads(cleaned_text), "API_CALL"
    except Exception as e:
        return [{"error": str(e)}], "ERROR"

def process_single_row(args):
    """处理单行数据的函数，供 ThreadPoolExecutor 调用"""
    index, row, resources, exact_match_lookup, reviewed_lookup, canonical_question_text, preloaded_sub_index, original_indices_for_sub_index, stats_lock = args
    current_text = row['text']
    
    # 第零层: 已审核结果
    reviewed_key = (BATCH_QUESTION_TEXT, current_text)
    if reviewed_key in reviewed_lookup:
        return {"original_index": index, "process_method": "REVIEWED_CORRECTED", "coding_results": reviewed_lookup[reviewed_key]}
        
    # 第一层: 无意义内容
    if is_meaningless(current_text):
        return {"original_index": index, "process_method": "PREFILTERED_MEANINGLESS", "coding_results": MEANINGLESS_CODE_JSON}

    # 第二层: 知识库精准匹配
    lookup_key = (canonical_question_text, current_text)
    if lookup_key in exact_match_lookup:
        return {"original_index": index, "process_method": "KB_EXACT_MATCH", "coding_results": exact_match_lookup[lookup_key]}
        
    # 第三层 & 第四层: 相似度匹配或API调用
    coding_results_list, call_type = rag_pipeline_final(
        resources, current_text, SIMILARITY_COPY_THRESHOLD,
        preloaded_sub_index, original_indices_for_sub_index
    )
    return {"original_index": index, "process_method": call_type.upper(), "coding_results": coding_results_list}

# ==============================================================================
# --- 主执行区 ---
# ==============================================================================
if __name__ == "__main__":
    try:
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    except ImportError:
        print("错误: 缺少tenacity库。请运行: pip install tenacity")
        exit()

    try:
        print("--- 步骤 1: 正在加载所有共享资源 ---")
        resources = {
            "df_map": pd.read_pickle(DATA_MAP_FILE),
            "codebook_df": pd.read_csv(CODEBOOK_FILE, dtype={'code': str}).set_index('code'),
            "answer_index": faiss.read_index(ANSWER_FAISS_INDEX_FILE),
        }
        uncoded_df = pd.read_csv(BATCH_INPUT_FILE, dtype=str).fillna('')
        print("共享资源加载成功！")
    except Exception as e:
        print(f"错误：启动时加载共享资源失败: {e}。")
        exit()

    print(f"--- 步骤 2: 为批处理问题“{BATCH_QUESTION_TEXT[:30]}...”准备上下文 ---")
    question_index = faiss.read_index(QUESTION_FAISS_INDEX_FILE)
    with open(UNIQUE_QUESTIONS_FILE, 'rb') as f: unique_questions = pickle.load(f)
    with open(QUESTION_TO_SUB_INDEX_MAP_FILE, 'rb') as f: question_to_sub_index_map = pickle.load(f)
    best_match_q_idx = find_best_matching_question_index(question_index, unique_questions, BATCH_QUESTION_TEXT)
    canonical_question_text = unique_questions[best_match_q_idx]
    print(f"智能匹配到最相似的历史问题: “{canonical_question_text}”")
    sub_index_info = question_to_sub_index_map[canonical_question_text]
    preloaded_sub_index = faiss.read_index(sub_index_info["index_file"])
    original_indices_for_sub_index = sub_index_info["original_indices"]
    print("专属迷你索引信息已加载。")

    print("--- 步骤 3: 正在构建知识库精准匹配查找表 ---")
    exact_match_lookup = {}
    grouped = resources["df_map"].groupby(['question', 'text'])['code'].apply(lambda codes: sorted(list(set(codes.astype(str)))))
    for (question, text), codes in tqdm(grouped.items(), desc="Building Exact Match Lookup"):
        key = (question, text)
        code_info_list = []
        for code in codes:
            try:
                code_series = resources["codebook_df"].loc[code]
                code_info = clean_code_info(code_series, code)
                code_info_list.append(code_info)
            except KeyError: continue
        if code_info_list: exact_match_lookup[key] = code_info_list
    print(f"精准匹配查找表构建完成，包含 {len(exact_match_lookup)} 条记录。")
    
    print("--- 步骤 3.5: 正在构建已校正编码查找表 ---")
    reviewed_lookup = {}
    if os.path.exists(REVIEWED_RESULTS_FOLDER):
        reviewed_files = glob.glob(os.path.join(REVIEWED_RESULTS_FOLDER, "**", "*_CORRECTED.jsonl"), recursive=True)
        if reviewed_files:
            print(f"找到 {len(reviewed_files)} 个已审核结果文件，正在加载...")
            for filepath in reviewed_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                q, t, results = record.get("question"), record.get("text"), record.get("coding_results")
                                if q and t and results is not None:
                                    reviewed_lookup[(q, t)] = results
                            except json.JSONDecodeError: continue
    print(f"已校正编码查找表构建完成，包含 {len(reviewed_lookup)} 条记录。")
    
    # --- 步骤 4: 正在进行数据清洗与批次内去重 ---
    uncoded_df['text'] = uncoded_df['text'].str.strip()
    uncoded_df = uncoded_df[uncoded_df['text'] != ''].reset_index(drop=True)
    uncoded_df.reset_index(inplace=True, drop=False) # 使用 'index' 列作为原始顺序的唯一标识

    # 首先创建只包含不重复文本的DataFrame
    unique_texts_df = uncoded_df.drop_duplicates(subset=['text']).copy()

    # 【【修复】】然后从这个不重复的DataFrame来创建正确的映射表
    # 这样可以保证 key 是文本, value 是该文本【首次】出现的原始索引
    first_occurrence_map = pd.Series(unique_texts_df['index'].values, index=unique_texts_df.text).to_dict()

    print(f"批次内去重完成，总共 {len(uncoded_df)} 条记录，其中 {len(unique_texts_df)} 条为不重复记录需要处理。")
    
    stats = {"total": len(uncoded_df), "prefiltered_hits": 0, "kb_exact_hits": 0, "similarity_hits": 0, "api_calls": 0, "errors": 0, "reviewed_hits": 0}
    stats_lock = Lock()
    
    tasks = [(row['index'], row, resources, exact_match_lookup, reviewed_lookup, canonical_question_text, preloaded_sub_index, original_indices_for_sub_index, stats_lock) for _, row in unique_texts_df.iterrows()]
    
    output_results = [None] * len(uncoded_df)

    print(f"--- 步骤 5: 开始多线程编码 {len(tasks)} 条不重复数据 (线程数: {NUM_THREADS}) ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_original_index = {executor.submit(process_single_row, task): task[0] for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_original_index), total=len(tasks), desc="Coding Unique Items"):
            original_index = future_to_original_index[future]
            try:
                result_data = future.result()
                output_results[original_index] = result_data
            except Exception as e:
                output_results[original_index] = {"original_index": original_index, "process_method": "MAIN_THREAD_ERROR", "coding_results": [{"error": str(e)}]}

    print("--- 步骤 6: 正在为批次内重复数据复制结果 ---")
    for idx, row in tqdm(uncoded_df.iterrows(), total=len(uncoded_df), desc="Copying duplicate results"):
        if output_results[idx] is None: # 这是一个重复项
            first_idx = first_occurrence_map.get(row['text'])
            if first_idx is not None and output_results[first_idx] is not None:
                source_result = output_results[first_idx]
                copied_result = source_result.copy()
                copied_result['process_method'] = f"BATCH_DUPLICATE_COPY"
                output_results[idx] = copied_result

    print(f"\n--- 步骤 7: 正在保存已处理的 {len(output_results)} 条记录... ---")
    final_output_records = []
    
    # 在主线程中一次性更新统计数据
    for result_data in output_results:
        if result_data:
            method = result_data.get('process_method', 'UNKNOWN').upper()
            if method == "REVIEWED_CORRECTED": stats['reviewed_hits'] += 1
            elif method == "PREFILTERED_MEANINGLESS": stats['prefiltered_hits'] += 1
            elif method == "KB_EXACT_MATCH": stats['kb_exact_hits'] += 1
            elif method == "SIMILARITY_MATCH_COPY": stats['similarity_hits'] += 1
            elif method == "API_CALL": stats['api_calls'] += 1
            elif "ERROR" in method: stats['errors'] += 1
            
    for idx, result_data in enumerate(output_results):
        if result_data:
            row = uncoded_df.iloc[idx]
            final_record = {"uuid": row.get('uuid', ''), "question": BATCH_QUESTION_TEXT, "text": row['text'], **result_data}
            if 'original_index' in final_record: del final_record['original_index']
            final_output_records.append(final_record)
            
    safe_batch_name = re.sub(r'[\\/*?:"<>|]', "", BATCH_QUESTION_TEXT)[:50]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(RESULTS_FOLDER, f"coded_batch_{safe_batch_name}_{timestamp}.jsonl")
    with open(output_filename, 'w', encoding='utf-8') as f:
        for record in final_output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"结果已保存至 {output_filename}")
    print("-" * 50)
    print("本次任务运行统计:")
    stats['total_processed_this_run'] = len(final_output_records)
    print(f"本次运行总处理条数: {stats['total_processed_this_run']}")
    print(f"  - 预过滤命中: {stats['prefiltered_hits']}")
    print(f"  - 知识库精准匹配命中: {stats['kb_exact_hits']}")
    print(f"  - 相似度匹配命中: {stats['similarity_hits']}")
    print(f"  - 已校正数据直接使用: {stats['reviewed_hits']}")
    print(f"  - 批次内重复数据去重: {len(uncoded_df) - len(unique_texts_df)}")
    print(f"  ------------------------------------")
    print(f"  - 实际API调用次数: {stats['api_calls']}")
    print(f"  - 处理失败条数: {stats['errors']}")
    print("-" * 50)