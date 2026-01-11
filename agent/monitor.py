"""
可观测性模块

Token统计、延迟监控、错误追踪、Fallback机制
使用 SQLite 存储
"""
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger


class Monitor:
    """监控器"""

    FALLBACK_RESPONSES = [
        "稍等一下，我确认一下再回复你～",
        "不好意思，系统有点忙，稍后回复你",
        "让我想想，稍等哈～",
    ]

    def __init__(self, db_path: str = "data/chat_history.db"):
        self.db_path = db_path
        self._current_call: Optional[Dict] = None
        self._fallback_index = 0
        self._db = None

    def _get_db(self):
        if self._db is None:
            from storage import get_database
            self._db = get_database(self.db_path)
        return self._db

    def start_call(self, thread_id: str, stage: str = ""):
        """开始记录一次调用"""
        self._current_call = {
            "start_time": time.time(),
            "thread_id": thread_id,
            "stage": stage,
            "tools_called": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def record_tokens(self, input_tokens: int, output_tokens: int):
        """记录 Token 用量"""
        if self._current_call:
            self._current_call["input_tokens"] = input_tokens
            self._current_call["output_tokens"] = output_tokens

    def record_tool_call(self, tool_name: str):
        """记录工具调用"""
        if self._current_call:
            self._current_call["tools_called"].append(tool_name)

    def end_call(self, success: bool = True, error: str = None):
        """结束记录，保存到数据库"""
        if not self._current_call:
            return

        latency_ms = (time.time() - self._current_call["start_time"]) * 1000
        input_tokens = self._current_call.get("input_tokens", 0)
        output_tokens = self._current_call.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        try:
            self._get_db().save_metrics(
                timestamp=datetime.now().isoformat(),
                thread_id=self._current_call.get("thread_id", ""),
                stage=self._current_call.get("stage", ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
                success=success,
                error=error,
                tools_called=self._current_call.get("tools_called", [])
            )
        except Exception as e:
            logger.warning(f"保存指标失败: {e}")

        logger.info(f"调用完成: tokens={total_tokens}, latency={latency_ms:.0f}ms, success={success}")
        self._current_call = None

    def get_fallback_response(self) -> str:
        """获取 Fallback 回复"""
        response = self.FALLBACK_RESPONSES[self._fallback_index]
        self._fallback_index = (self._fallback_index + 1) % len(self.FALLBACK_RESPONSES)
        return response

    def get_stats(self, date: str = None) -> Dict[str, Any]:
        """获取统计数据"""
        return self._get_db().get_metrics_stats(date)


# 全局实例
_monitor: Optional[Monitor] = None


def get_monitor(db_path: str = "data/chat_history.db") -> Monitor:
    """获取监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = Monitor(db_path)
    return _monitor
