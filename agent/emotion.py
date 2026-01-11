"""
情绪分析模块

使用 Erlangshen-Roberta-110M-Sentiment 模型进行情感分析
"""
import os
import sys
import warnings
import logging

# 抑制TensorFlow和protobuf警告
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 抑制protobuf的AttributeError输出
class _SuppressProtobufErrors:
    def __init__(self, stream):
        self._stream = stream
    def write(self, msg):
        if "MessageFactory" not in msg and "GetPrototype" not in msg:
            self._stream.write(msg)
    def flush(self):
        self._stream.flush()

sys.stderr = _SuppressProtobufErrors(sys.stderr)

from typing import Dict, Optional
from loguru import logger

# 尝试导入 transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers 未安装，将使用规则引擎进行情绪分析")

# 模型配置（京东电商评论情感模型，对电商场景理解更好）
EMOTION_MODEL = "uer/roberta-base-finetuned-jd-binary-chinese"
USE_EMOTION_MODEL = os.getenv("USE_EMOTION_MODEL", "true").lower() == "true"
# 置信度阈值：低于此值判为neutral（模型只支持positive/negative二分类）
NEUTRAL_THRESHOLD = float(os.getenv("EMOTION_NEUTRAL_THRESHOLD", "0.83"))
# 模型保存目录（通过环境变量设置，transformers会自动使用）
MODEL_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR", r"D:\develop\huggingface")
# 设置环境变量，让transformers使用指定的缓存目录
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # 强制离线模式


class EmotionAnalyzer:
    """情绪分析器（基于 Erlangshen-Roberta 模型）"""
    
    def __init__(self, use_model: bool = None):
        """
        Args:
            use_model: 是否使用模型（None则从环境变量读取）
        """
        self.use_model = use_model if use_model is not None else USE_EMOTION_MODEL
        self.analyzer = None
        
        if self.use_model and TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"加载情绪分析模型: {EMOTION_MODEL}")
                logger.info(f"模型保存目录: {MODEL_CACHE_DIR}")
                # 使用环境变量设置缓存目录，pipeline会自动使用
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model=EMOTION_MODEL,
                    tokenizer=EMOTION_MODEL,
                    device=-1  # CPU，如果有GPU可以改为0
                )
                logger.info("情绪分析模型加载成功")
            except Exception as e:
                logger.error(f"情绪分析模型加载失败: {e}，将使用规则引擎")
                self.analyzer = None
                self.use_model = False
    
    def analyze(self, text: str, context: str = "") -> Dict[str, any]:
        """分析情绪

        Args:
            text: 当前消息
            context: 上下文（可选）

        Returns:
            {
                "sentiment": "positive|negative|neutral",
                "confidence": 0.0-1.0,
                "method": "model|rule"
            }
        """
        # 合并上下文和当前消息
        full_text = f"{context}\n{text}" if context else text

        # 使用模型分析
        if self.analyzer:
            try:
                # pipeline 的调用方式（直接传入文本即可）
                result = self.analyzer(full_text)

                # 模型返回格式: [{"label": "positive/negative", "score": 0.95}]
                if result and len(result) > 0:
                    label = result[0].get("label", "").lower()
                    score = result[0].get("score", 0.5)

                    # 映射到我们的格式（京东模型标签：label_0=negative, label_1=positive）
                    if score < NEUTRAL_THRESHOLD:
                        # 置信度低于阈值，判为neutral
                        sentiment = "neutral"
                    elif "positive" in label or label == "label_1":
                        sentiment = "positive"
                    elif "negative" in label or label == "label_0":
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

                    return {
                        "sentiment": sentiment,
                        "confidence": float(score),
                        "method": "model"
                    }
            except Exception as e:
                logger.warning(f"模型分析失败: {e}，降级为规则引擎")

        # 降级：使用规则引擎
        return self._analyze_by_rule(text)

    def _analyze_by_rule(self, text: str) -> Dict[str, any]:
        """规则引擎分析（降级方案）"""
        text_lower = text.lower()

        # 情绪判断
        positive_keywords = ["好", "可以", "行", "ok", "不错", "满意", "谢谢"]
        negative_keywords = ["贵", "太贵", "便宜", "不行", "不好", "算了", "不要"]

        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "confidence": 0.6,  # 规则引擎置信度较低
            "method": "rule"
        }


# 全局实例（懒加载）
_emotion_analyzer: Optional[EmotionAnalyzer] = None


def get_emotion_analyzer() -> EmotionAnalyzer:
    """获取情绪分析器实例（单例）"""
    global _emotion_analyzer
    if _emotion_analyzer is None:
        _emotion_analyzer = EmotionAnalyzer()
    return _emotion_analyzer

