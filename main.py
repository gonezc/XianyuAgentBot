"""
闲鱼自动客服 - 主入口

基于 LangGraph 的智能客服系统
"""
import asyncio
import os
import sys
from loguru import logger
from dotenv import load_dotenv

from XianyuApis import XianyuApis
from storage import get_database
from agent import create_workflow
from core import XianyuLive


def setup_logging():
    """配置日志"""
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.info(f"日志级别: {log_level}")


def main():
    load_dotenv()
    setup_logging()

    cookies_str = os.getenv("COOKIES_STR")
    if not cookies_str:
        logger.error("请设置 COOKIES_STR 环境变量")
        sys.exit(1)

    db_path = "data/chat_history.db"

    # 初始化组件
    xianyu_api = XianyuApis()
    db = get_database(db_path)
    agent_graph = create_workflow(memory_type="sqlite", db_path=db_path)
    logger.info("Agent 初始化完成")

    # 启动客户端
    client = XianyuLive(cookies_str, agent_graph, db, xianyu_api)
    asyncio.run(client.main())


if __name__ == '__main__':
    main()
