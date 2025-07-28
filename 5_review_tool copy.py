# 5_review_tool.py
import streamlit as st
import os
import pandas as pd
from utils.file_utils import (
    load_codebook,
    list_raw_batches,
    list_corrected_batches,
    get_batch_name,
    get_corrected_path,
    get_raw_path_from_corrected
)
from utils.review_manager import ReviewManager
from utils.logger import ReviewLogger
from utils.backup_manager import BackupManager

def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    if 'logger' not in st.session_state:
        st.session_state.logger = ReviewLogger()
    if 'backup_manager' not in st.session_state:
        st.session_state.backup_manager = BackupManager()
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "reviewer_001"

def get_file_status():
    """获取文件状态分类 - 改进版"""
    raw_files = list_raw_batches("05_coded_results")
    corrected_files = list_corrected_batches("06_reviewed_results")

    unstarted = []
    ongoing = []
    finished = []

    unstarted.extend(raw_files)

    for corrected_file in corrected_files:
        corresponding_raw_file = get_raw_path_from_corrected(corrected_file, "05_coded_results", "06_reviewed_results")

        if corresponding_raw_file and os.path.exists(corresponding_raw_file):
            if corresponding_raw_file in unstarted:
                unstarted.remove(corresponding_raw_file)

            try:
                with open(corresponding_raw_file, 'r', encoding='utf-8') as f:
                    raw_count = sum(1 for _ in f)
                with open(corrected_file, 'r', encoding='utf-8') as f:
                    corrected_count = sum(1 for _ in f)

                if corrected_count >= raw_count:
                    finished.append(corresponding_raw_file)
                else:
                    ongoing.append(corresponding_raw_file)
            except Exception as e:
                print(f"计算 {corresponding_raw_file} 进度时出错: {e}")
                ongoing.append(corresponding_raw_file)
        else:
            print(f"警告: 无法找到修正文件 {corrected_file} 对应的原始文件，或原始文件不存在。")

    return unstarted, ongoing, finished

def show_home_page():
    st.title("🤖 AI 编码人工审核平台")
    st.markdown("---")

    st.session_state.logger.log_action(st.session_state.user_name, "访问首页")

    unstarted, ongoing, finished = get_file_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📝 待审核 - 未开始")
        if unstarted:
            selected_unstarted = st.selectbox(
                "选择批次",
                unstarted,
                format_func=lambda x: os.path.relpath(x, '05_coded_results'),
                key="home_unstarted_select"
            )
            if st.button("🚀 开始审核", key="start_review_btn"):
                if selected_unstarted:
                    st.session_state.selected_batch = selected_unstarted
                    st.session_state.current_page = "review"
                    st.session_state.logger.log_file_access(
                        st.session_state.user_name,
                        os.path.relpath(selected_unstarted, '05_coded_results'),
                        "开始审核"
                    )
                    # 清理可能残留的旧 session state
                    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(('review_manager', 'edited_codes', 'current_uuid_for_editing', 'current_', 'show_return_confirm', 'backup_created'))]
                    for k in keys_to_delete:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
                else:
                    st.warning("请先选择一个批次")
        else:
            st.info("暂无未开始的批次")

    with col2:
        st.subheader("🔄 待审核 - 进行中")
        if ongoing:
            selected_ongoing = st.selectbox(
                "继续审核",
                ongoing,
                format_func=lambda x: os.path.relpath(x, '05_coded_results'),
                key="home_ongoing_select"
            )
            raw_file = selected_ongoing
            corrected_file = get_corrected_path(raw_file)
            if os.path.exists(raw_file) and os.path.exists(corrected_file):
                try:
                    with open(raw_file, 'r', encoding='utf-8') as f:
                        raw_count = sum(1 for _ in f)
                    with open(corrected_file, 'r', encoding='utf-8') as f:
                        corrected_count = sum(1 for _ in f)
                    progress = corrected_count / raw_count if raw_count > 0 else 0
                    st.progress(progress)
                    st.caption(f"进度: {corrected_count}/{raw_count} ({progress*100:.1f}%)")
                except Exception as e:
                    st.error(f"计算进度出错: {e}")

            if st.button("▶️ 继续", key="continue_review_btn"):
                if selected_ongoing:
                    st.session_state.selected_batch = selected_ongoing
                    st.session_state.current_page = "review"
                    st.session_state.logger.log_file_access(
                        st.session_state.user_name,
                        os.path.relpath(selected_ongoing, '05_coded_results'),
                        "继续审核"
                    )
                    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(('review_manager', 'edited_codes', 'current_uuid_for_editing', 'current_', 'show_return_confirm', 'backup_created'))]
                    for k in keys_to_delete:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
                else:
                    st.warning("请先选择一个批次")
        else:
            st.info("暂无进行中的批次")

    with col3:
        st.subheader("✅ 已完成")
        if finished:
            selected_finished = st.selectbox(
                "查看历史",
                finished,
                format_func=lambda x: os.path.relpath(x, '05_coded_results'),
                key="home_finished_select"
            )
            if st.button("👁️ 查看/编辑", key="view_finished_btn"):
                if selected_finished:
                    st.session_state.selected_batch = selected_finished
                    st.session_state.current_page = "review"
                    st.session_state.logger.log_file_access(
                        st.session_state.user_name,
                        os.path.relpath(selected_finished, '05_coded_results'),
                        "查看已完成"
                    )
                    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(('review_manager', 'edited_codes', 'current_uuid_for_editing', 'current_', 'show_return_confirm', 'backup_created'))]
                    for k in keys_to_delete:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
                else:
                    st.warning("请先选择一个批次")
        else:
            st.info("暂无已完成的批次")

    st.markdown("---")
    st.subheader("📊 系统统计")
    st.write(f"总批次数量: {len(unstarted) + len(ongoing) + len(finished)}")
    st.write(f"未开始: {len(unstarted)} | 进行中: {len(ongoing)} | 已完成: {len(finished)}")

def show_review_page():
    # --- 新增：注入自定义CSS以减小字体和元素尺寸，并添加 sticky header 样式 ---
    st.markdown("""
    <style>
    /* 减小全局字体大小 */
    html, body, [class*="css"] {
        font-size: 12px; /* 默认通常是 16px */
    }
    /* 减小标题字体大小 */
    .stApp h1 { font-size: 1.6rem !important; } /* 主标题 */
    .stApp h2 { font-size: 1.3rem !important; } /* 子标题 */
    .stApp h3 { font-size: 1.1rem !important; }
    .stApp h4 { font-size: 1.0rem !important; }
    .stApp h5 { font-size: 0.9rem !important; }
    .stApp h6 { font-size: 0.8rem !important; }

    /* 减小 selectbox 和 multiselect 的高度 */
    .stSelectbox div[data-baseweb="select"] > div {
        min-height: 26px !important; /* 默认可能是 38px 或更高 */
        font-size: 11px !important;
    }
    .stMultiSelect div[data-baseweb="select"] > div {
        min-height: 26px !important;
        font-size: 11px !important;
    }

    /* 减小按钮的内边距 */
    .stButton button {
        padding: 0.15rem 0.3rem !important; /* 默认可能是 0.5rem 1rem */
        font-size: 10px !important;
        height: 1.6rem !important; /* 控制按钮高度 */
    }

    /* 减小 text_input 的高度 */
    .stTextInput div[data-baseweb="input"] > input {
        min-height: 26px !important;
        font-size: 11px !important;
    }

    /* 减小 info 和 warning 等容器的内边距 */
    .stAlert, .stInfo, .stWarning, .stSuccess, .stError {
        padding: 0.3rem 0.6rem !important; /* 默认可能是 1rem 1.5rem */
    }

    /* 减小 expander 的内边距 */
    .streamlit-expanderHeader {
        padding: 0.15rem 0.5rem !important;
        font-size: 11px !important;
    }
    .streamlit-expanderContent {
        padding: 0.3rem 0.6rem !important;
    }

    /* 减小 progress bar 的高度 */
    .stProgress > div > div > div {
        min-height: 0.2rem !important; /* 默认可能是 0.4rem */
    }
    
    /* 减小 markdown 文本的默认边距 */
    .element-container .stMarkdown {
        margin-bottom: 0.2rem !important;
    }
    
    /* 减小 hr 标签的边距 */
    hr {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* --- 新增：使原始信息区域在滚动时固定在顶部 --- */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 100; /* 确保它在其他元素之上 */
        background-color: white; /* 防止滚动时内容透过 */
        padding: 6px 0; /* 调整内边距 */
        border-bottom: 1px solid #e0e0e0; /* 可选：添加底部分隔线 */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* 可选：添加轻微阴影增强视觉效果 */
        font-size: 12px;
    }
    .sticky-header strong {
        font-size: 12px;
    }
    /* --- 结束新增 --- */
    </style>
    """, unsafe_allow_html=True)
    # --- 结束新增/修改 CSS ---

    def confirm_return():
        st.session_state.show_return_confirm = True

    def cancel_return():
        st.session_state.show_return_confirm = False

    def execute_return():
        st.session_state.current_page = "home"
        st.session_state.show_return_confirm = False
        # 清理 session state
        keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(('review_manager', 'edited_codes', 'current_uuid_for_editing', 'current_', 'backup_created'))]
        for k in keys_to_delete:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    if st.session_state.get('show_return_confirm', False):
        st.warning("⚠️ 确认返回首页？未保存的修改将会丢失！")
        col1, col2 = st.columns(2)
        with col1:
            st.button("✅ 确认返回", on_click=execute_return)
        with col2:
            st.button("❌ 取消", on_click=cancel_return)
        return

    codebook_df = load_codebook("01_source_data/last_phase_codebook.csv")

    if 'review_manager' not in st.session_state:
        st.session_state.review_manager = ReviewManager(
            st.session_state.selected_batch,
            codebook_df,
            st.session_state.logger
        )

    manager = st.session_state.review_manager

    if 'backup_created' not in st.session_state:
        backup_path = st.session_state.backup_manager.create_backup(manager.corrected_file)
        if backup_path:
            st.session_state.logger.log_action(
                st.session_state.user_name,
                "创建备份",
                f"备份文件: {os.path.basename(backup_path)}"
            )
        st.session_state.backup_created = True

    current_item = manager.get_current_item()
    if not current_item:
        st.error("无法获取当前审核项")
        return

    # --- 修改：确保 current_item 和 corrected_item 都有 'coding_results' 字段 ---
    if 'coding_results' not in current_item or not isinstance(current_item['coding_results'], list):
        current_item['coding_results'] = []

    corrected_item = manager.get_corrected_item(current_item['uuid'])
    if corrected_item is None:
        corrected_item = current_item.copy()
    if 'coding_results' not in corrected_item or not isinstance(corrected_item['coding_results'], list):
        corrected_item['coding_results'] = []
    # --- 结束修改 ---

    # 为了 UI 一致性，创建一个临时的 'codes' 字段用于前端展示
    current_item_for_ui = current_item.copy()
    current_item_for_ui['codes'] = current_item['coding_results']
    corrected_item_for_ui = corrected_item.copy()
    corrected_item_for_ui['codes'] = corrected_item['coding_results']

    progress_info = manager.get_progress_info()
    st.progress(progress_info['percentage'] / 100)
    st.caption(f"进度: {progress_info['completed']}/{progress_info['total']} ({progress_info['percentage']:.1f}%)")

    # 管理按钮栏 (使用更紧凑的布局)
    st.markdown("---")
    btn_col1, btn_col_spacer1, btn_col2, btn_col_spacer2, btn_col3, btn_col_spacer3, btn_col4, btn_col_spacer4, btn_col5 = st.columns([2, 0.2, 2, 0.2, 2, 0.2, 2, 0.2, 2])

    with btn_col1:
        if st.button("🏠 << 返回首页", key="return_home_btn"):
            confirm_return()

    with btn_col2:
        if st.button("⬅️ < 上一条", key="prev_item_btn", disabled=(manager.get_current_index() == 0)):
            manager.prev_item()
            prev_item = manager.get_current_item()
            if prev_item:
                if 'coding_results' not in prev_item or not isinstance(prev_item['coding_results'], list):
                    prev_item['coding_results'] = []
                prev_corrected = manager.get_corrected_item(prev_item['uuid']) or prev_item.copy()
                if 'coding_results' not in prev_corrected or not isinstance(prev_corrected['coding_results'], list):
                    prev_corrected['coding_results'] = []
                st.session_state.edited_codes = prev_corrected['coding_results'].copy()
                st.session_state.current_uuid_for_editing = prev_item['uuid']
                keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith('current_') and prev_item['uuid'] not in k]
                for k in keys_to_delete:
                    if k in st.session_state:
                        del st.session_state[k]
            st.rerun()

    with btn_col3:
        item_to_save = corrected_item_for_ui.copy()
        item_to_save['coding_results'] = item_to_save.pop('codes', [])
        if st.button("💾 仅保存", key="save_only_btn"):
            if manager.save_current_item(item_to_save):
                st.success("✅ 保存成功")
                st.session_state.logger.log_action(
                    st.session_state.user_name,
                    "保存记录",
                    f"UUID: {item_to_save.get('uuid', 'unknown')}"
                )
            else:
                st.error("❌ 保存失败")

    with btn_col4:
        if st.button("↩️ 撤销修改", key="reset_item_btn"):
            st.session_state.edited_codes = current_item_for_ui.get('codes', []).copy()
            st.session_state.current_uuid_for_editing = current_item['uuid']
            keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith('current_') and current_item['uuid'] in k]
            for k in keys_to_delete:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("✅ 已恢复为AI原始编码")
            st.session_state.logger.log_action(
                st.session_state.user_name,
                "撤销修改",
                f"UUID: {current_item['uuid']}"
            )
            st.rerun()

    with btn_col5:
        item_to_save_next = corrected_item_for_ui.copy()
        item_to_save_next['coding_results'] = item_to_save_next.pop('codes', [])
        if st.button("➡️ 保存并下一条 >", key="save_next_btn", disabled=(manager.get_current_index() == manager.get_total_count() - 1)):
            if manager.save_current_item(item_to_save_next):
                manager.next_item()
                st.session_state.logger.log_action(
                    st.session_state.user_name,
                    "保存并切换",
                    f"UUID: {item_to_save_next.get('uuid', 'unknown')}"
                )
                next_item = manager.get_current_item()
                if next_item:
                    if 'coding_results' not in next_item or not isinstance(next_item['coding_results'], list):
                        next_item['coding_results'] = []
                    next_corrected = manager.get_corrected_item(next_item['uuid']) or next_item.copy()
                    if 'coding_results' not in next_corrected or not isinstance(next_corrected['coding_results'], list):
                        next_corrected['coding_results'] = []
                    st.session_state.edited_codes = next_corrected['coding_results'].copy()
                    st.session_state.current_uuid_for_editing = next_item['uuid']
                    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith('current_') and next_item['uuid'] not in k]
                    for k in keys_to_delete:
                        if k in st.session_state:
                            del st.session_state[k]
                st.rerun()
            else:
                st.error("❌ 保存失败")

    # --- 修改：使用 sticky container 显示原始信息 ---
    header_container = st.container()
    question_text = current_item.get('question', '')
    answer_text = current_item.get('text', '')
    header_container.markdown(
        f"""
        <div class="sticky-header">
            <strong>问题:</strong> {question_text} &nbsp;&nbsp;&nbsp;&nbsp;
            <strong>回答:</strong> {answer_text}
        </div>
        """,
        unsafe_allow_html=True
    )
    # --- 结束修改 ---

    # 注意：ReviewManager 仍然使用码表的 codes 选项，UI 层需要适配
    code_options = manager.code_options

    # 初始化或更新 session state 中的 edited_codes
    if 'edited_codes' not in st.session_state or st.session_state.get('current_uuid_for_editing') != current_item['uuid']:
        st.session_state.edited_codes = corrected_item_for_ui.get('codes', []).copy()
        st.session_state.current_uuid_for_editing = current_item['uuid']

    # --- 修改：实现基于 sentiment 的联动筛选 (无刷新) ---
    for i, code_item in enumerate(st.session_state.edited_codes):
        col_sent, col_net, col_subnet, col_label, col_code, col_del = st.columns([1, 2, 2, 3, 1, 0.5])

        # --- 联动逻辑实现 (无页面刷新) ---
        base_key = f"current_{i}_{current_item['uuid']}"

        current_sentiment = st.session_state.get(f"{base_key}_sentiment", code_item.get('sentiment', ''))
        current_net = st.session_state.get(f"{base_key}_net", code_item.get('net', ''))
        current_subnet = st.session_state.get(f"{base_key}_subnet", code_item.get('subnet', ''))
        current_label = st.session_state.get(f"{base_key}_label", code_item.get('label', ''))
        current_code = st.session_state.get(f"{base_key}_code", str(code_item.get('code', '')))

        def get_options_for_field(field_name, current_value, sentiment_val, net_val, subnet_val):
            """根据上游字段值和当前值，获取指定字段的选项列表"""
            opts = []
            df = manager.codebook_df
            if sentiment_val:
                df = df[df['sentiment'] == sentiment_val]
            if net_val:
                df = df[df['net'] == net_val]
            if subnet_val is not None:
                df = df[df['subnet'] == subnet_val]
            
            if not df.empty:
                if field_name == 'code':
                    opts = sorted(df[field_name].dropna().unique().tolist(), key=lambda x: (int(x) if str(x).isdigit() else float('inf'), str(x)))
                else:
                    opts = sorted(df[field_name].dropna().unique().tolist())
            
            if current_value and current_value not in opts:
                opts.insert(0, current_value)
            
            if field_name == 'code' and 'x' not in opts:
                opts.insert(0, 'x')
            
            if field_name == 'subnet' and '' not in opts:
                opts.insert(0, '')
                 
            return opts

        # 渲染 sentiment 下拉框
        with col_sent:
            sent_opts = get_options_for_field('sentiment', current_sentiment, None, None, None)
            selected_sentiment = st.selectbox(
                f"S{i}",
                options=sent_opts,
                index=sent_opts.index(current_sentiment) if current_sentiment in sent_opts else 0,
                key=f"{base_key}_sentiment_sel",
                help="情感"
            )
            if selected_sentiment != current_sentiment:
                st.session_state[f"{base_key}_sentiment"] = selected_sentiment
                st.session_state.edited_codes[i]['sentiment'] = selected_sentiment
                net_opts = get_options_for_field('net', '', selected_sentiment, None, None)
                first_net = next((opt for opt in net_opts if opt), '')
                st.session_state[f"{base_key}_net"] = first_net
                st.session_state.edited_codes[i]['net'] = first_net
                subnet_opts = get_options_for_field('subnet', '', selected_sentiment, first_net, None)
                first_subnet = next((opt for opt in subnet_opts if opt is not None), '')
                st.session_state[f"{base_key}_subnet"] = first_subnet
                st.session_state.edited_codes[i]['subnet'] = first_subnet
                label_opts = get_options_for_field('label', '', selected_sentiment, first_net, first_subnet)
                first_label = next((opt for opt in label_opts if opt), '')
                st.session_state[f"{base_key}_label"] = first_label
                st.session_state.edited_codes[i]['label'] = first_label
                code_opts = get_options_for_field('code', '', selected_sentiment, first_net, first_subnet)
                first_code = next((opt for opt in code_opts if opt), '')
                st.session_state[f"{base_key}_code"] = first_code
                st.session_state.edited_codes[i]['code'] = first_code

        # 渲染 net 下拉框
        with col_net:
            current_sent_for_net = st.session_state.get(f"{base_key}_sentiment", current_sentiment)
            net_opts = get_options_for_field('net', current_net, current_sent_for_net, None, None)
            selected_net = st.selectbox(
                f"N{i}",
                options=net_opts,
                index=net_opts.index(current_net) if current_net in net_opts else 0,
                key=f"{base_key}_net_sel",
                help="网络"
            )
            if selected_net != current_net:
                st.session_state[f"{base_key}_net"] = selected_net
                st.session_state.edited_codes[i]['net'] = selected_net
                subnet_opts = get_options_for_field('subnet', '', current_sent_for_net, selected_net, None)
                first_subnet = next((opt for opt in subnet_opts if opt is not None), '')
                st.session_state[f"{base_key}_subnet"] = first_subnet
                st.session_state.edited_codes[i]['subnet'] = first_subnet
                label_opts = get_options_for_field('label', '', current_sent_for_net, selected_net, first_subnet)
                first_label = next((opt for opt in label_opts if opt), '')
                st.session_state[f"{base_key}_label"] = first_label
                st.session_state.edited_codes[i]['label'] = first_label
                code_opts = get_options_for_field('code', '', current_sent_for_net, selected_net, first_subnet)
                first_code = next((opt for opt in code_opts if opt), '')
                st.session_state[f"{base_key}_code"] = first_code
                st.session_state.edited_codes[i]['code'] = first_code

        # 渲染 subnet 下拉框
        with col_subnet:
            current_sent_for_subnet = st.session_state.get(f"{base_key}_sentiment", current_sentiment)
            current_net_for_subnet = st.session_state.get(f"{base_key}_net", current_net)
            subnet_opts = get_options_for_field('subnet', current_subnet, current_sent_for_subnet, current_net_for_subnet, None)
            selected_subnet = st.selectbox(
                f"SN{i}",
                options=subnet_opts,
                index=subnet_opts.index(current_subnet) if current_subnet in subnet_opts else 0,
                key=f"{base_key}_subnet_sel",
                help="子网络"
            )
            if selected_subnet != current_subnet:
                st.session_state[f"{base_key}_subnet"] = selected_subnet
                st.session_state.edited_codes[i]['subnet'] = selected_subnet
                label_opts = get_options_for_field('label', '', current_sent_for_subnet, current_net_for_subnet, selected_subnet)
                first_label = next((opt for opt in label_opts if opt), '')
                st.session_state[f"{base_key}_label"] = first_label
                st.session_state.edited_codes[i]['label'] = first_label
                code_opts = get_options_for_field('code', '', current_sent_for_subnet, current_net_for_subnet, selected_subnet)
                first_code = next((opt for opt in code_opts if opt), '')
                st.session_state[f"{base_key}_code"] = first_code
                st.session_state.edited_codes[i]['code'] = first_code

        # 渲染 label 下拉框
        with col_label:
            current_sent_for_label = st.session_state.get(f"{base_key}_sentiment", current_sentiment)
            current_net_for_label = st.session_state.get(f"{base_key}_net", current_net)
            current_subnet_for_label = st.session_state.get(f"{base_key}_subnet", current_subnet)
            label_opts = get_options_for_field('label', current_label, current_sent_for_label, current_net_for_label, current_subnet_for_label)
            selected_label = st.selectbox(
                f"L{i}",
                options=label_opts,
                index=label_opts.index(current_label) if current_label in label_opts else 0,
                key=f"{base_key}_label_sel",
                help="标签"
            )
            if selected_label != current_label:
                st.session_state[f"{base_key}_label"] = selected_label
                st.session_state.edited_codes[i]['label'] = selected_label
                df_filtered = manager.codebook_df[
                    (manager.codebook_df['sentiment'] == current_sent_for_label) &
                    (manager.codebook_df['net'] == current_net_for_label) &
                    (manager.codebook_df['subnet'] == current_subnet_for_label) &
                    (manager.codebook_df['label'] == selected_label)
                ]
                matching_codes = df_filtered['code'].dropna().unique().tolist()
                if matching_codes:
                    first_code = str(matching_codes[0])
                else:
                    code_opts = get_options_for_field('code', '', current_sent_for_label, current_net_for_label, current_subnet_for_label)
                    first_code = next((opt for opt in code_opts if opt), '')
                st.session_state[f"{base_key}_code"] = first_code
                st.session_state.edited_codes[i]['code'] = first_code

        # 渲染 code 下拉框
        with col_code:
            current_sent_for_code = st.session_state.get(f"{base_key}_sentiment", current_sentiment)
            current_net_for_code = st.session_state.get(f"{base_key}_net", current_net)
            current_subnet_for_code = st.session_state.get(f"{base_key}_subnet", current_subnet)
            code_opts = get_options_for_field('code', current_code, current_sent_for_code, current_net_for_code, current_subnet_for_code)
            selected_code = st.selectbox(
                f"C{i}",
                options=code_opts,
                index=code_opts.index(current_code) if current_code in code_opts else 0,
                key=f"{base_key}_code_sel",
                help="编码"
            )
            if selected_code != current_code:
                st.session_state[f"{base_key}_code"] = selected_code
                st.session_state.edited_codes[i]['code'] = selected_code

        # --- 结束联动逻辑 ---

        with col_del:
             if st.button("🗑️", key=f"delete_{i}_{current_item['uuid']}_btn", help="删除此行"):
                 st.session_state.edited_codes.pop(i)
                 keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(f"current_{i}_{current_item['uuid']}")]
                 for k in keys_to_delete:
                     if k in st.session_state:
                         del st.session_state[k]
                 st.rerun()

        st.markdown("---")

    # 更新修正项的 codes 部分 (UI层)
    corrected_item_for_ui['codes'] = st.session_state.edited_codes

    # 添加新编码按钮
    if st.button("➕ 添加新编码", key="add_new_code_btn"):
        st.session_state.edited_codes.append({
            'code': '',
            'sentiment': '',
            'net': '',
            'subnet': '',
            'label': ''
        })
        new_index = len(st.session_state.edited_codes) - 1
        new_base_key = f"current_{new_index}_{current_item['uuid']}"
        st.session_state[f"{new_base_key}_sentiment"] = ''
        st.session_state[f"{new_base_key}_net"] = ''
        st.session_state[f"{new_base_key}_subnet"] = ''
        st.session_state[f"{new_base_key}_label"] = ''
        st.session_state[f"{new_base_key}_code"] = ''
        st.rerun()

    # 快速跳转
    st.markdown("---")
    st.subheader("🔍 快速跳转")
    jump_to = st.number_input(
        "跳转到第几条",
        min_value=1,
        max_value=manager.get_total_count(),
        value=manager.get_current_index() + 1,
        key="jump_to_input"
    )
    if st.button("🚀 跳转", key="jump_button"):
        if manager.goto_item(jump_to - 1):
            jumped_item = manager.get_current_item()
            if jumped_item:
                if 'coding_results' not in jumped_item or not isinstance(jumped_item['coding_results'], list):
                    jumped_item['coding_results'] = []
                jumped_corrected = manager.get_corrected_item(jumped_item['uuid']) or jumped_item.copy()
                if 'coding_results' not in jumped_corrected or not isinstance(jumped_corrected['coding_results'], list):
                    jumped_corrected['coding_results'] = []
                st.session_state.edited_codes = jumped_corrected['coding_results'].copy()
                st.session_state.current_uuid_for_editing = jumped_item['uuid']
            st.rerun()
        else:
            st.error("跳转失败")

def main():
    init_session_state()

    st.set_page_config(
        page_title="AI编码审核工具",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    if st.session_state.current_page == "home":
        show_home_page()
    elif st.session_state.current_page == "review":
        show_review_page()

if __name__ == "__main__":
    main()