"""
数据库存储层 - 精简版
"""
import os
import sqlite3
import json
from datetime import datetime
from loguru import logger


class Database:
    """数据库存储服务"""

    def __init__(self, db_path: str = "data/chat_history.db", max_history: int = 100):
        self.db_path = db_path
        self.max_history = max_history
        self._ensure_dir()
        self._init_tables()

    def _ensure_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        # 统一消息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT,
                strategy TEXT,
                stage TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_thread_id ON messages(thread_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)')

        # 对话元数据表 (合并 handover_status, chat_bargain_counts, conversation_stats)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT,
                item_id TEXT,
                bargain_count INTEGER DEFAULT 0,
                is_handover INTEGER DEFAULT 0,
                handover_time DATETIME,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                total_rounds INTEGER DEFAULT 0,
                stage_reached TEXT,
                is_deal INTEGER DEFAULT 0,
                deal_price REAL
            )
        ''')

        # 商品信息表 (去掉冗余字段)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 调用指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS call_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                thread_id TEXT,
                stage TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0,
                success INTEGER DEFAULT 1,
                error TEXT,
                tools_called TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON call_metrics(timestamp)')

        conn.commit()
        conn.close()
        logger.info(f"数据库初始化完成: {self.db_path}")

    # ========== 对话线程管理 ==========

    def _ensure_thread(self, thread_id: str, user_id: str = None, item_id: str = None):
        """确保线程存在"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO threads (thread_id, user_id, item_id) VALUES (?, ?, ?)",
            (thread_id, user_id, item_id)
        )
        conn.commit()
        conn.close()

    # ========== 消息管理 ==========

    def add_message_by_chat(self, chat_id: str, user_id: str, item_id: str, role: str, content: str):
        """添加消息"""
        self._ensure_thread(chat_id, user_id, item_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content)
        )
        # 清理旧消息
        cursor.execute(
            "DELETE FROM messages WHERE thread_id = ? AND id NOT IN (SELECT id FROM messages WHERE thread_id = ? ORDER BY timestamp DESC LIMIT ?)",
            (chat_id, chat_id, self.max_history)
        )
        conn.commit()
        conn.close()

    def get_context_by_chat(self, chat_id: str) -> list:
        """获取对话历史"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY timestamp ASC LIMIT ?",
            (chat_id, self.max_history)
        )
        messages = [{"role": role, "content": content} for role, content in cursor.fetchall()]
        bargain_count = self.get_bargain_count(chat_id)
        if bargain_count > 0:
            messages.append({"role": "system", "content": f"议价次数: {bargain_count}"})
        conn.close()
        return messages

    def save_message(self, thread_id: str, role: str, content: str, item_desc: str = "",
                     emotion: dict = None, strategy: str = "", stage: str = ""):
        """保存消息（带元数据）"""
        self._ensure_thread(thread_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (thread_id, role, content, emotion, strategy, stage) VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, role, content, str(emotion) if emotion else "", strategy, stage)
        )
        conn.commit()
        conn.close()

    def get_history(self, thread_id: str, limit: int = 50) -> list:
        """获取历史记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, timestamp FROM messages WHERE thread_id = ? ORDER BY timestamp DESC LIMIT ?",
            (thread_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        return list(reversed(rows))

    # ========== 议价管理 ==========

    def increment_bargain_count_by_chat(self, chat_id: str):
        self._ensure_thread(chat_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE threads SET bargain_count = bargain_count + 1 WHERE thread_id = ?", (chat_id,))
        conn.commit()
        conn.close()

    def get_bargain_count(self, thread_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT bargain_count FROM threads WHERE thread_id = ?", (thread_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0

    def get_bargain_count_by_chat(self, chat_id: str) -> int:
        return self.get_bargain_count(chat_id)

    # ========== 商品管理 ==========

    def save_item_info(self, item_id: str, item_data: dict):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO items (item_id, data, last_updated) VALUES (?, ?, ?) ON CONFLICT(item_id) DO UPDATE SET data = ?, last_updated = ?",
            (item_id, json.dumps(item_data, ensure_ascii=False), datetime.now().isoformat(),
             json.dumps(item_data, ensure_ascii=False), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    def get_item_info(self, item_id: str) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM items WHERE item_id = ?", (item_id,))
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else None

    # ========== 转人工状态 ==========

    def is_handover(self, thread_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT is_handover FROM threads WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()
        conn.close()
        return row is not None and row[0] == 1

    def set_handover(self, thread_id: str, is_handover: bool = True):
        self._ensure_thread(thread_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE threads SET is_handover = ?, handover_time = ? WHERE thread_id = ?",
            (1 if is_handover else 0, datetime.now().isoformat(), thread_id)
        )
        conn.commit()
        conn.close()

    def clear_handover(self, thread_id: str):
        self.set_handover(thread_id, False)

    # ========== 调用指标 ==========

    def save_metrics(self, timestamp: str, thread_id: str, stage: str,
                     input_tokens: int, output_tokens: int, total_tokens: int,
                     latency_ms: float, success: bool, error: str, tools_called: list):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO call_metrics (timestamp, thread_id, stage, input_tokens, output_tokens, total_tokens, latency_ms, success, error, tools_called) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, thread_id, stage, input_tokens, output_tokens, total_tokens, latency_ms, 1 if success else 0, error, json.dumps(tools_called))
        )
        conn.commit()
        conn.close()

    def get_metrics_stats(self, date: str = None) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        query = "SELECT COUNT(*), SUM(total_tokens), AVG(latency_ms), SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) FROM call_metrics"
        if date:
            cursor.execute(query + " WHERE timestamp LIKE ?", (f"{date}%",))
        else:
            cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        total_calls = row[0] or 0
        return {
            "total_calls": total_calls,
            "total_tokens": row[1] or 0,
            "avg_latency_ms": round(row[2] or 0, 2),
            "total_errors": row[3] or 0,
            "error_rate": round((row[3] or 0) / total_calls * 100, 2) if total_calls > 0 else 0
        }

    # ========== 对话统计 ==========

    def update_conversation_stats(self, thread_id: str, total_rounds: int = None,
                                   stage_reached: str = None, bargain_count: int = None):
        self._ensure_thread(thread_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        if total_rounds is not None:
            cursor.execute("UPDATE threads SET total_rounds = ? WHERE thread_id = ?", (total_rounds, thread_id))
        if stage_reached is not None:
            cursor.execute("UPDATE threads SET stage_reached = ? WHERE thread_id = ?", (stage_reached, thread_id))
        if bargain_count is not None:
            cursor.execute("UPDATE threads SET bargain_count = ? WHERE thread_id = ?", (bargain_count, thread_id))
        conn.commit()
        conn.close()

    def record_deal(self, thread_id: str, price: float):
        self._ensure_thread(thread_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE threads SET is_deal = 1, deal_price = ?, end_time = ? WHERE thread_id = ?",
            (price, datetime.now().isoformat(), thread_id)
        )
        conn.commit()
        conn.close()

    def get_daily_stats(self, date: str = None) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        query = "SELECT COUNT(*), SUM(is_deal), SUM(deal_price) FROM threads"
        if date:
            cursor.execute(query + " WHERE start_time LIKE ?", (f"{date}%",))
        else:
            cursor.execute(query)
        row = cursor.fetchone()
        total, deals, revenue = row[0] or 0, row[1] or 0, row[2] or 0
        cursor.execute("SELECT stage_reached, COUNT(*) FROM threads GROUP BY stage_reached")
        stage_dist = {r[0]: r[1] for r in cursor.fetchall() if r[0]}
        conn.close()
        return {
            "date": date or "all",
            "total_conversations": total,
            "deals": deals,
            "deal_rate": round(deals / total * 100, 2) if total > 0 else 0,
            "total_revenue": revenue,
            "stage_distribution": stage_dist
        }

    # 兼容旧接口
    def save_conversation_stats(self, thread_id: str, start_time: str, end_time: str,
                                 total_rounds: int, stage_reached: str, is_deal: bool,
                                 deal_price: float, bargain_count: int):
        self._ensure_thread(thread_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE threads SET start_time=?, end_time=?, total_rounds=?, stage_reached=?, is_deal=?, deal_price=?, bargain_count=? WHERE thread_id=?",
            (start_time, end_time, total_rounds, stage_reached, 1 if is_deal else 0, deal_price, bargain_count, thread_id)
        )
        conn.commit()
        conn.close()


# 全局实例
_db = None


def get_database(db_path: str = "data/chat_history.db") -> Database:
    global _db
    if _db is None:
        _db = Database(db_path)
    return _db
