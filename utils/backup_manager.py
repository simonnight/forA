# utils/backup_manager.py
import os
import shutil
from datetime import datetime
import glob

class BackupManager:
    def __init__(self, backup_dir="backups", max_backups=10):
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        os.makedirs(backup_dir, exist_ok=True)
    
    def create_backup(self, file_path):
        """为指定文件创建备份"""
        if not os.path.exists(file_path):
            return None
            
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        
        # 生成备份文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        # 执行备份
        try:
            shutil.copy2(file_path, backup_path)
            
            # 清理旧备份
            self._cleanup_old_backups(name)
            
            return backup_path
        except Exception as e:
            print(f"创建备份失败: {e}")
            return None
    
    def _cleanup_old_backups(self, base_name):
        """清理超过最大数量的旧备份"""
        pattern = os.path.join(self.backup_dir, f"{base_name}_backup_*.jsonl")
        backup_files = sorted(glob.glob(pattern))
        
        # 如果备份文件超过最大数量，删除最旧的
        while len(backup_files) > self.max_backups:
            oldest_backup = backup_files.pop(0)
            try:
                os.remove(oldest_backup)
            except OSError as e:
                print(f"删除旧备份失败: {e}")
    
    def get_backup_list(self, base_filename):
        """获取指定文件的所有备份列表"""
        pattern = os.path.join(self.backup_dir, f"{base_filename}_backup_*.jsonl")
        return sorted(glob.glob(pattern), reverse=True)
    
    def restore_backup(self, backup_path, target_path):
        """从备份恢复文件"""
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, target_path)
                return True
            except Exception as e:
                print(f"恢复备份失败: {e}")
                return False
        return False
    
    def get_backup_info(self, backup_path):
        """获取备份文件信息"""
        if os.path.exists(backup_path):
            stat = os.stat(backup_path)
            return {
                'path': backup_path,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            }
        return None