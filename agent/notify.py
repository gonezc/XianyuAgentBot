"""
通知服务

飞书通知：转人工、准备下单
"""
import os
import json
import requests
from loguru import logger


def _send_feishu_card(message: str, notice_type: str = "reminder") -> dict:
    """发送飞书卡片通知

    Args:
        message: 通知内容（Markdown格式）
        notice_type: 通知类型，"reminder"(准备下单) 或 "handover"(转人工)

    Returns:
        dict: {"success": bool, "message": str}
    """
    webhook_url = os.getenv("FEISHU_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("飞书 Webhook 未配置，跳过通知")
        return {"success": True, "message": "通知已记录（Webhook未配置）"}

    # 根据通知类型设置样式
    if notice_type == "handover":
        template = "orange"
        title = "需要转人工"
    else:
        template = "yellow"
        title = "准备接单"

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": template
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": message}}
            ]
        }
    }

    # 转人工通知添加提示
    if notice_type == "handover":
        card["card"]["elements"].append({
            "tag": "note",
            "elements": [{"tag": "plain_text", "content": "请及时跟进，可能需要人工定价或确认需求"}]
        })

    try:
        resp = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(card),
            timeout=10
        )
        if resp.status_code == 200 and resp.json().get("code") == 0:
            logger.info(f"飞书通知发送成功: {message[:30]}...")
            return {"success": True, "message": "通知已发送"}
        return {"success": False, "message": resp.json().get("msg", "发送失败")}
    except Exception as e:
        logger.error(f"飞书通知异常: {e}")
        return {"success": False, "message": str(e)}
