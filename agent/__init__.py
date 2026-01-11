"""
闲鱼AI客服Agent

基于LangGraph的智能客服工作流
"""
import os
import sys
import warnings
import logging

# 抑制TensorFlow和protobuf警告（必须在导入其他模块之前）
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# 抑制protobuf的AttributeError输出
class _SuppressProtobufErrors:
    def __init__(self, stream):
        self._stream = stream
    def write(self, msg):
        # 过滤protobuf相关的错误信息
        if msg.strip() in (":", "", "AttributeError"):
            return
        if "MessageFactory" in msg or "GetPrototype" in msg or "AttributeError" in msg:
            return
        self._stream.write(msg)
    def flush(self):
        self._stream.flush()

sys.stderr = _SuppressProtobufErrors(sys.stderr)

from .graph import create_workflow, process_message, AgentState
from .tools import tools, search_cases, send_reminder
from .knowledge import KnowledgeBase

__all__ = [
    "create_workflow",
    "process_message",
    "AgentState",
    "tools",
    "search_cases",
    "send_reminder",
    "KnowledgeBase"
]
