"""
工具定义

LLM 可调用的工具（Function Calling）
"""
import os
import json
import requests
from langchain_core.tools import tool
from loguru import logger

from .knowledge import KnowledgeBase

# 知识库实例
_kb = None

def _get_kb():
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


# ============ 工具定义 ============

@tool
def search_cases(query: str, top_k: int = 3) -> str:
    """【重要】搜索相似案例，获取参考价格和工期
    
    何时调用：
    - 当需要给用户报价时，必须先调用此工具获取参考价格
    - 当用户问"多少钱"、"报价"时调用
    - 当需要展示类似项目经验时调用
    
    Args:
        query: 项目关键词，如"图书管理系统"、"Spring Boot后台"、"爬虫采集"
        top_k: 返回结果数量，默认3条
        
    Returns:
        JSON格式的案例列表，包含：
        - title: 案例标题
        - price: 参考价格（元）
        - duration: 预计工期
        - tags: 技术标签
        - _score: 相似度分数
        
    示例返回：
        [{"title": "图书馆借阅系统", "price": 3200, "duration": "15天", ...}]
    """
    results = _get_kb().search(query, top_k)
    
    # 没有匹配结果或相似度很低
    if not results:
        logger.info(f"搜索案例 '{query}': 无匹配结果")
        return json.dumps({
            "found": False,
            "message": "没有找到匹配的案例，这个需求比较特殊，建议加微信详聊定价"
        }, ensure_ascii=False)
    
    # 检查相似度：如果最高分低于阈值，说明需求不明确
    max_score = max(r.get("_score", 0) for r in results) if results else 0
    if max_score < 0.5:  # 相似度阈值
        logger.info(f"搜索案例 '{query}': 相似度较低 (最高分={max_score:.3f})，需求可能不明确")
        return json.dumps({
            "found": False,
            "message": f"找到的案例相似度较低（最高{max_score:.2f}），这个需求比较特殊，建议加微信详聊定价"
        }, ensure_ascii=False)
    
    # 简化返回，只保留关键信息
    simplified = []
    for r in results:
        simplified.append({
            "title": r.get("title", ""),
            "price": r.get("price", 0),
            "duration": r.get("duration", ""),
            "tags": r.get("tags", [])[:3],  # 只取前3个标签
            "_score": round(r.get("_score", 0), 2)
        })
    
    logger.info(f"搜索案例 '{query}': {len(results)} 条, 价格范围: {[r['price'] for r in simplified]}")
    return json.dumps({
        "found": True,
        "cases": simplified,
        "price_range": [min(r["price"] for r in simplified), max(r["price"] for r in simplified)]
    }, ensure_ascii=False, indent=2)

@tool
def send_reminder(message: str, notice_type: str = "reminder") -> str:
    """发送飞书通知

    何时调用：
    - 价格谈妥，客户表示会下单时（notice_type="reminder"）
    - 客户说"行"、"可以"、"我去下单"等确认意向时（notice_type="reminder"）
    - search_cases 返回 found=false 时，需要转人工（notice_type="handover"）

    Args:
        message: 通知内容，如"图书系统 2000元 客户说去下单了"
        notice_type: 通知类型，"reminder"(准备下单) 或 "handover"(转人工)

    Returns:
        发送结果
    """
    from .notify import _send_feishu_card
    result = _send_feishu_card(message, notice_type=notice_type)
    return json.dumps(result, ensure_ascii=False)


# 导出工具列表（LLM可调用的）
tools = [search_cases, send_reminder]
