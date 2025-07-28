# utils/review_manager.py
import os
from utils.data_loader import load_jsonl, append_to_jsonl
from utils.file_utils import get_corrected_path

class ReviewManager:
    def __init__(self, raw_file, codebook_df, logger=None):
        self.raw_file = raw_file
        self.corrected_file = get_corrected_path(raw_file) # 使用更新后的函数
        self.codebook_df = codebook_df
        self.logger = logger
        
        # 加载数据
        self.raw_data = load_jsonl(raw_file)
        self.corrected_data = load_jsonl(self.corrected_file) if os.path.exists(self.corrected_file) else []
        
        # 创建UUID映射
        self.uuid_map = {item['uuid']: item for item in self.corrected_data}
        self.current_index = 0
        
        # 获取码表选项
        self.code_options = self._get_code_options()
    
    def _get_code_options(self):
        """从码表获取所有选项，处理空值"""
        options = {
            'code': [],
            'sentiment': [],
            'net': [],
            'subnet': [], # 需要处理空值
            'label': []
        }
        
        if not self.codebook_df.empty:
            for col in options.keys():
                if col in self.codebook_df.columns:
                    # 提取唯一值，转换为字符串，处理 NaN/None
                    # dropna=False 保留空值，fillna('') 将 NaN 替换为空字符串
                    # astype(str) 确保所有值是字符串
                    unique_vals = self.codebook_df[col].fillna('').astype(str).unique().tolist()
                    # 过滤掉 'nan' 字符串（虽然fillna应该已经处理了）
                    unique_vals = [v for v in unique_vals if v.lower() != 'nan']
                    # 排序，code 列按数值排序
                    if col == 'code':
                        try:
                            # 尝试按数值排序
                            options[col] = sorted(unique_vals, key=lambda x: (x == '', int(x) if x.isdigit() else float('inf'), x))
                        except ValueError:
                            # 如果转换失败，按字符串排序，空字符串放前面
                            options[col] = sorted(unique_vals, key=lambda x: (x == '', x))
                    else:
                        # 其他列按字符串排序，空字符串放前面
                        options[col] = sorted(unique_vals, key=lambda x: (x == '', x))
        # print("提取的选项:", options) # 调试用
        return options
    
    # ... (其余方法保持不变或根据需要微调) ...
    # 注意：get_progress_info, get_current_item, get_corrected_item, 
    # save_current_item, next_item, prev_item, goto_item, 
    # get_current_index, get_total_count, reset_to_original 等方法
    # 应该保持不变或只需确保使用了 self.corrected_file 等正确路径
    def get_progress_info(self):
        """获取进度信息"""
        total = len(self.raw_data)
        completed = len(self.corrected_data)
        return {
            'total': total,
            'completed': completed,
            'remaining': total - completed,
            'percentage': (completed / total * 100) if total > 0 else 0
        }
    
    def get_current_item(self):
        """获取当前项"""
        if 0 <= self.current_index < len(self.raw_data):
            return self.raw_data[self.current_index]
        return None

    def get_corrected_item(self, uuid):
        """获取已修正的项"""
        return self.uuid_map.get(uuid, None)

    def save_current_item(self, corrected_item):
        """保存当前项"""
        try:
            # 记录日志
            if self.logger:
                original_item = self.get_corrected_item(corrected_item['uuid'])
                if original_item:
                    self.logger.log_data_change(
                        self.logger.user_name, 
                        corrected_item['uuid'], 
                        "update", 
                        f"Updated codes from {len(original_item.get('coding_results', []))} to {len(corrected_item.get('coding_results', []))} items"
                    )
                else:
                    self.logger.log_data_change(
                        "reviewer_001", 
                        corrected_item['uuid'], 
                        "create", 
                        f"Created new correction with {len(corrected_item.get('codes', []))} codes"
                    )
            
            # 保存到内存映射
            self.uuid_map[corrected_item['uuid']] = corrected_item
            
            # 追加到文件
            return append_to_jsonl(self.corrected_file, corrected_item)
        except Exception as e:
            if self.logger:
                self.logger.log_error("reviewer_001", f"保存失败: {str(e)}")
            print(f"保存失败: {e}") # 添加打印以便调试
            return False

    def next_item(self):
        """下一项"""
        if self.current_index < len(self.raw_data) - 1:
            self.current_index += 1
            return True
        return False

    def prev_item(self):
        """上一项"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def goto_item(self, index):
        """跳转到指定项"""
        if 0 <= index < len(self.raw_data):
            self.current_index = index
            return True
        return False

    def get_current_index(self):
        """获取当前索引"""
        return self.current_index

    def get_total_count(self):
        """获取总数"""
        return len(self.raw_data)

    def reset_to_original(self, uuid):
        """重置为原始AI编码"""
        # 找到原始项
        for item in self.raw_data:
            if item['uuid'] == uuid:
                # 从修正映射中移除
                if uuid in self.uuid_map:
                    del self.uuid_map[uuid]
                return item.copy()
        return None
