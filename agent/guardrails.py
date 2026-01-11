"""
安全护栏模块

输入/输出过滤、敏感词检测、越狱防护
"""
import re
from typing import Tuple, List, Optional
from loguru import logger


class Guardrails:
    """安全护栏"""

    # 输入过滤：检测越狱/注入攻击
    INPUT_PATTERNS = [
        r"忽略.*指令",
        r"忘记.*设定",
        r"你现在是",
        r"假装你是",
        r"不要遵守",
        r"ignore.*instructions",
        r"forget.*rules",
        r"you are now",
        r"pretend to be",
        r"jailbreak",
        r"DAN",
    ]

    # 输出过滤：AI不应该说的话
    OUTPUT_FORBIDDEN = [
        # 承诺无法兑现的事
        r"保证.*天.*完成",
        r"一定能.*通过",
        r"100%",
        r"绝对没问题",
        # 泄露定价策略
        r"底价是",
        r"成本只有",
        r"利润.*%",
        # 不当言论
        r"傻[逼比]",
        r"垃圾",
        r"滚",
    ]

    # 敏感词替换
    SENSITIVE_REPLACE = {
        r"加我?微信[号]?[\s:：]*\S+": "加微信详聊（微信号见商品页）",
        r"QQ[号]?[\s:：]*\d+": "QQ详聊（QQ号见商品页）",
        r"手机[号]?[\s:：]*1\d{10}": "电话详聊（电话见商品页）",
    }

    def __init__(self):
        self.input_patterns = [re.compile(p, re.IGNORECASE) for p in self.INPUT_PATTERNS]
        self.output_patterns = [re.compile(p, re.IGNORECASE) for p in self.OUTPUT_FORBIDDEN]
        self.sensitive_patterns = [(re.compile(k, re.IGNORECASE), v) for k, v in self.SENSITIVE_REPLACE.items()]

    def check_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查用户输入

        Returns:
            (is_safe, reason): 是否安全，不安全时返回原因
        """
        for pattern in self.input_patterns:
            if pattern.search(text):
                logger.warning(f"输入触发安全规则: {pattern.pattern}")
                return False, "detected_injection"
        return True, None

    def check_output(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查AI输出

        Returns:
            (is_safe, reason): 是否安全，不安全时返回原因
        """
        for pattern in self.output_patterns:
            if pattern.search(text):
                logger.warning(f"输出触发安全规则: {pattern.pattern}")
                return False, "forbidden_content"
        return True, None

    def sanitize_output(self, text: str) -> str:
        """清理AI输出，替换敏感信息"""
        result = text
        for pattern, replacement in self.sensitive_patterns:
            result = pattern.sub(replacement, result)
        return result

    def process_input(self, text: str) -> Tuple[str, bool]:
        """处理用户输入

        Returns:
            (processed_text, should_respond): 处理后的文本，是否应该回复
        """
        is_safe, reason = self.check_input(text)
        if not is_safe:
            logger.warning(f"用户输入被拦截: {reason}")
            return text, False
        return text, True

    def process_output(self, text: str) -> str:
        """处理AI输出

        Returns:
            处理后的文本（清理敏感信息，拦截违规内容）
        """
        is_safe, reason = self.check_output(text)
        if not is_safe:
            logger.warning(f"AI输出被拦截: {reason}")
            return "不好意思，我需要确认一下再回复你～"
        return self.sanitize_output(text)


# 全局实例
_guardrails: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    """获取安全护栏实例"""
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails
