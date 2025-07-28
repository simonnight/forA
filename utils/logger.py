# utils/logger.py
import logging
import os
from datetime import datetime

class ReviewLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名（按日期）
        log_filename = os.path.join(log_dir, f"review_log_{datetime.now().strftime('%Y%m%d')}.log")
        
        # 配置日志器
        self.logger = logging.getLogger("ReviewTool")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            handler = logging.FileHandler(log_filename, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_action(self, user, action, details=""):
        """记录用户操作"""
        message = f"User: {user} | Action: {action}"
        if details:
            message += f" | Details: {details}"
        self.logger.info(message)
    
    def log_error(self, user, error_msg):
        """记录错误信息"""
        self.logger.error(f"User: {user} | Error: {error_msg}")
    
    def log_file_access(self, user, filename, operation):
        """记录文件访问"""
        self.log_action(user, f"{operation} file", filename)
    
    def log_data_change(self, user, uuid, change_type, details=""):
        """记录数据变更"""
        message = f"User: {user} | UUID: {uuid} | Change: {change_type}"
        if details:
            message += f" | Details: {details}"
        self.logger.info(message)