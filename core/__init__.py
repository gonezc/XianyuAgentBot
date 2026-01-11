"""
核心模块

包含 WebSocket 客户端、消息处理、人工接管等核心功能
"""
from .websocket_client import XianyuLive
from .message_handler import MessageHandler
from .heartbeat import HeartbeatManager

__all__ = ["XianyuLive", "MessageHandler", "HeartbeatManager"]
