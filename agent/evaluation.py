"""
评估模块

成交率统计、对话效果分析
使用 SQLite 存储
"""
from typing import Dict, Any, Optional
from loguru import logger


class Evaluator:
    """评估器"""

    def __init__(self, db_path: str = "data/chat_history.db"):
        self.db_path = db_path
        self._db = None
        self._round_counts: Dict[str, int] = {}  # 内存中跟踪轮数

    def _get_db(self):
        if self._db is None:
            from storage import get_database
            self._db = get_database(self.db_path)
        return self._db

    def update_conversation(self, thread_id: str, stage: str = None, bargain_count: int = None):
        """更新对话状态"""
        # 更新轮数
        self._round_counts[thread_id] = self._round_counts.get(thread_id, 0) + 1

        try:
            self._get_db().update_conversation_stats(
                thread_id=thread_id,
                total_rounds=self._round_counts[thread_id],
                stage_reached=stage,
                bargain_count=bargain_count
            )
        except Exception as e:
            logger.warning(f"更新对话统计失败: {e}")

    def record_deal(self, thread_id: str, price: float):
        """记录成交"""
        try:
            self._get_db().record_deal(thread_id, price)
            logger.info(f"成交记录: thread={thread_id}, price={price}")
        except Exception as e:
            logger.warning(f"记录成交失败: {e}")

    def get_daily_stats(self, date: str = None) -> Dict[str, Any]:
        """获取每日统计"""
        return self._get_db().get_daily_stats(date)


# 全局实例
_evaluator: Optional[Evaluator] = None


def get_evaluator(db_path: str = "data/chat_history.db") -> Evaluator:
    """获取评估器实例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator(db_path)
    return _evaluator
