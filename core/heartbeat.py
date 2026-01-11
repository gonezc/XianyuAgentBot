"""
心跳管理模块
"""
import asyncio
import json
import time
from loguru import logger
from utils.xianyu_utils import generate_mid


class HeartbeatManager:
    """WebSocket 心跳管理"""

    def __init__(self, interval: int = 15, timeout: int = 5):
        self.interval = interval
        self.timeout = timeout
        self.last_heartbeat_time = 0
        self.last_heartbeat_response = 0
        self.task = None

    async def send(self, ws) -> str:
        """发送心跳包"""
        heartbeat_mid = generate_mid()
        msg = {"lwp": "/!", "headers": {"mid": heartbeat_mid}}
        await ws.send(json.dumps(msg))
        self.last_heartbeat_time = time.time()
        logger.debug("心跳包已发送")
        return heartbeat_mid

    async def loop(self, ws):
        """心跳维护循环"""
        while True:
            try:
                current = time.time()
                if current - self.last_heartbeat_time >= self.interval:
                    await self.send(ws)
                if (current - self.last_heartbeat_response) > (self.interval + self.timeout):
                    logger.warning("心跳响应超时")
                    break
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"心跳循环出错: {e}")
                break

    def handle_response(self, message_data: dict) -> bool:
        """处理心跳响应，返回是否为心跳响应"""
        if (isinstance(message_data, dict)
            and "headers" in message_data
            and "mid" in message_data["headers"]
            and message_data.get("code") == 200):
            self.last_heartbeat_response = time.time()
            logger.debug("收到心跳响应")
            return True
        return False

    def start(self, ws):
        """启动心跳任务"""
        self.last_heartbeat_time = time.time()
        self.last_heartbeat_response = time.time()
        self.task = asyncio.create_task(self.loop(ws))
        return self.task

    async def stop(self):
        """停止心跳任务"""
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
