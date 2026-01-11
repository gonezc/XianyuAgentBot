"""
对话存储服务

兼容层：使用存储层
"""
from storage import get_database as _get_db, Database

# 兼容旧接口
ConversationStore = Database


def get_store(db_path: str = "data/chat_history.db") -> Database:
    """获取存储实例"""
    return _get_db(db_path)
