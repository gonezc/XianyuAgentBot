"""
消息处理模块
"""
import base64
import json
import time
import os
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from utils.xianyu_utils import generate_mid, generate_uuid, decrypt


class MessageHandler:
    """消息处理器"""

    def __init__(self, myid: str, context_manager, agent_graph, xianyu_api):
        self.myid = myid
        self.context_manager = context_manager
        self.agent_graph = agent_graph
        self.xianyu = xianyu_api

        # 人工接管
        self.manual_mode_conversations = set()
        self.manual_mode_timestamps = {}
        self.manual_mode_timeout = int(os.getenv("MANUAL_MODE_TIMEOUT", "3600"))
        self.toggle_keywords = os.getenv("TOGGLE_KEYWORDS", "。")

        # 消息过期时间
        self.message_expire_time = int(os.getenv("MESSAGE_EXPIRE_TIME", "300000"))

        # 线程池
        max_workers = int(os.getenv("THREAD_POOL_SIZE", str(min(32, (os.cpu_count() or 1) + 4))))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"初始化线程池，大小: {max_workers}")

    def is_chat_message(self, message: dict) -> bool:
        """判断是否为用户聊天消息"""
        try:
            return (isinstance(message, dict)
                    and "1" in message
                    and isinstance(message["1"], dict)
                    and "10" in message["1"]
                    and isinstance(message["1"]["10"], dict)
                    and "reminderContent" in message["1"]["10"])
        except Exception:
            return False

    def is_sync_package(self, message_data: dict) -> bool:
        """判断是否为同步包消息"""
        try:
            return (isinstance(message_data, dict)
                    and "body" in message_data
                    and "syncPushPackage" in message_data["body"]
                    and "data" in message_data["body"]["syncPushPackage"]
                    and len(message_data["body"]["syncPushPackage"]["data"]) > 0)
        except Exception:
            return False

    def is_typing_status(self, message: dict) -> bool:
        """判断是否为用户正在输入状态"""
        try:
            return (isinstance(message, dict)
                    and "1" in message
                    and isinstance(message["1"], list)
                    and len(message["1"]) > 0
                    and isinstance(message["1"][0], dict)
                    and "1" in message["1"][0]
                    and "@goofish" in message["1"][0].get("1", ""))
        except Exception:
            return False

    def is_system_message(self, message: dict) -> bool:
        """判断是否为系统消息"""
        try:
            return (isinstance(message, dict)
                    and "3" in message
                    and isinstance(message["3"], dict)
                    and message["3"].get("needPush") == "false")
        except Exception:
            return False

    def is_manual_mode(self, chat_id: str) -> bool:
        """检查是否处于人工接管模式"""
        if chat_id not in self.manual_mode_conversations:
            return False
        if chat_id in self.manual_mode_timestamps:
            if time.time() - self.manual_mode_timestamps[chat_id] > self.manual_mode_timeout:
                self.exit_manual_mode(chat_id)
                return False
        return True

    def enter_manual_mode(self, chat_id: str):
        self.manual_mode_conversations.add(chat_id)
        self.manual_mode_timestamps[chat_id] = time.time()

    def exit_manual_mode(self, chat_id: str):
        self.manual_mode_conversations.discard(chat_id)
        self.manual_mode_timestamps.pop(chat_id, None)

    def toggle_manual_mode(self, chat_id: str) -> str:
        if self.is_manual_mode(chat_id):
            self.exit_manual_mode(chat_id)
            return "auto"
        self.enter_manual_mode(chat_id)
        return "manual"

    def check_toggle_keywords(self, message: str) -> bool:
        return message.strip() in self.toggle_keywords

    def _send_order_notification(self, user_id: str, user_url: str, message: dict):
        """发送订单成交通知到飞书"""
        webhook_url = os.getenv("FEISHU_WEBHOOK_URL", "")
        if not webhook_url:
            return

        try:
            item_title = message.get('3', {}).get('title', '未知商品')
            price = message.get('3', {}).get('price', '未知')
        except:
            item_title, price = "未知商品", "未知"

        card = {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": "订单成交"}, "template": "green"},
                "elements": [
                    {"tag": "div", "fields": [
                        {"is_short": True, "text": {"tag": "lark_md", "content": f"**买家**\n[{user_id}]({user_url})"}},
                        {"is_short": True, "text": {"tag": "lark_md", "content": f"**金额**\n{price}"}}
                    ]},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**商品**\n{item_title}"}}
                ]
            }
        }

        try:
            resp = requests.post(webhook_url, headers={"Content-Type": "application/json"},
                               data=json.dumps(card), timeout=10)
            if resp.status_code == 200:
                logger.info(f"[飞书] 订单通知发送成功: {user_id}")
        except Exception as e:
            logger.error(f"[飞书] 订单通知异常: {e}")

    def _handle_order_message(self, message: dict) -> bool:
        """处理订单消息，返回是否已处理"""
        try:
            red_reminder = message.get('3', {}).get('redReminder', '')
            if not red_reminder:
                return False

            user_id = message['1'].split('@')[0]
            user_url = f'https://www.goofish.com/personal?userId={user_id}'

            if red_reminder == '等待买家付款':
                logger.info(f'[订单] 等待买家 {user_url} 付款')
                return True
            elif red_reminder == '交易关闭':
                logger.info(f'[订单] 买家 {user_id} 交易关闭')
                return True
            elif red_reminder == '等待卖家发货':
                logger.info(f'[订单] 交易成功！买家 {user_url} 已付款')
                self._send_order_notification(user_id, user_url, message)
                return True
        except Exception as e:
            logger.debug(f"订单消息解析: {e}")
        return False

    async def send_msg(self, ws, cid: str, toid: str, text: str):
        """发送消息"""
        text_data = {"contentType": 1, "text": {"text": text}}
        text_base64 = base64.b64encode(json.dumps(text_data).encode('utf-8')).decode('utf-8')
        msg = {
            "lwp": "/r/MessageSend/sendByReceiverScope",
            "headers": {"mid": generate_mid()},
            "body": [{
                "uuid": generate_uuid(),
                "cid": f"{cid}@goofish",
                "conversationType": 1,
                "content": {"contentType": 101, "custom": {"type": 1, "data": text_base64}},
                "redPointPolicy": 0,
                "extension": {"extJson": "{}"},
                "ctx": {"appVersion": "1.0", "platform": "web"},
                "mtags": {},
                "msgReadStatusSetting": 1
            }, {
                "actualReceivers": [f"{toid}@goofish", f"{self.myid}@goofish"]
            }]
        }
        await ws.send(json.dumps(msg))

    async def handle(self, message_data: dict, ws):
        """处理消息"""
        from agent import process_message

        # 发送 ACK
        try:
            if "headers" in message_data:
                ack = {
                    "code": 200,
                    "headers": {
                        "mid": message_data["headers"].get("mid", generate_mid()),
                        "sid": message_data["headers"].get("sid", "")
                    }
                }
                for key in ["app-key", "ua", "dt"]:
                    if key in message_data["headers"]:
                        ack["headers"][key] = message_data["headers"][key]
                await ws.send(json.dumps(ack))
        except Exception:
            pass

        if not self.is_sync_package(message_data):
            return

        # 解密数据
        sync_data = message_data["body"]["syncPushPackage"]["data"][0]
        if "data" not in sync_data:
            return

        try:
            data = sync_data["data"]
            try:
                data = base64.b64decode(data).decode("utf-8")
                json.loads(data)
                return  # 无需解密的消息
            except:
                message = json.loads(decrypt(data))
        except Exception as e:
            logger.error(f"消息解密失败: {e}")
            return

        # 处理订单消息
        if self._handle_order_message(message):
            return

        # 过滤非聊天消息
        if self.is_typing_status(message):
            return
        if not self.is_chat_message(message):
            logger.debug(f"非聊天消息: {message}")
            return

        # 解析聊天消息
        create_time = int(message["1"]["5"])
        send_user_name = message["1"]["10"]["reminderTitle"]
        send_user_id = message["1"]["10"]["senderUserId"]
        send_message = message["1"]["10"]["reminderContent"]

        # 过期消息
        if (time.time() * 1000 - create_time) > self.message_expire_time:
            logger.debug("过期消息丢弃")
            return

        url_info = message["1"]["10"]["reminderUrl"]
        item_id = url_info.split("itemId=")[1].split("&")[0] if "itemId=" in url_info else None
        chat_id = message["1"]["2"].split('@')[0]

        if not item_id:
            logger.warning("无法获取商品ID")
            return

        # 卖家控制命令
        if send_user_id == self.myid:
            if self.check_toggle_keywords(send_message):
                mode = self.toggle_manual_mode(chat_id)
                logger.info(f"{'已接管' if mode == 'manual' else '已恢复自动回复'} 会话 {chat_id}")
                return
            self.context_manager.add_message_by_chat(chat_id, self.myid, item_id, "assistant", send_message)
            logger.info(f"卖家人工回复: {send_message}")
            return

        logger.info(f"用户: {send_user_name}, 商品: {item_id}, 消息: {send_message}")
        self.context_manager.add_message_by_chat(chat_id, send_user_id, item_id, "user", send_message)

        # 人工接管模式
        if self.is_manual_mode(chat_id):
            logger.info(f"会话 {chat_id} 处于人工接管模式")
            return

        if self.is_system_message(message):
            return

        # 获取商品信息
        item_info = self.context_manager.get_item_info(item_id)
        if not item_info:
            api_result = self.xianyu.get_item_info(item_id)
            if 'data' in api_result and 'itemDO' in api_result['data']:
                item_info = api_result['data']['itemDO']
                self.context_manager.save_item_info(item_id, item_info)
            else:
                logger.warning(f"获取商品信息失败")
                return

        item_description = f"{item_info['desc']};当前商品售卖价格为:{item_info['soldPrice']}"

        # 调用 Agent
        loop = asyncio.get_event_loop()
        bot_reply = await loop.run_in_executor(
            self.executor, process_message,
            self.agent_graph, send_message, item_description,
            send_user_id, send_user_name, chat_id
        )

        logger.info(f"机器人回复: {bot_reply}")
        await self.send_msg(ws, chat_id, send_user_id, bot_reply)

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
