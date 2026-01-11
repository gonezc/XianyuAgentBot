"""
WebSocket 客户端
"""
import asyncio
import json
import time
import os
import websockets
from loguru import logger
from utils.xianyu_utils import trans_cookies, generate_device_id, generate_mid
from .heartbeat import HeartbeatManager
from .message_handler import MessageHandler


class XianyuLive:
    """闲鱼 WebSocket 客户端"""

    def __init__(self, cookies_str: str, agent_graph, context_manager, xianyu_api):
        self.base_url = 'wss://wss-goofish.dingtalk.com/'
        self.cookies_str = cookies_str
        self.cookies = trans_cookies(cookies_str)
        self.myid = self.cookies['unb']
        self.device_id = generate_device_id(self.myid)
        self.ws = None

        # 心跳管理
        self.heartbeat = HeartbeatManager(
            interval=int(os.getenv("HEARTBEAT_INTERVAL", "15")),
            timeout=int(os.getenv("HEARTBEAT_TIMEOUT", "5"))
        )

        # Token 管理
        self.token_refresh_interval = int(os.getenv("TOKEN_REFRESH_INTERVAL", "3600"))
        self.token_retry_interval = int(os.getenv("TOKEN_RETRY_INTERVAL", "300"))
        self.last_token_refresh_time = 0
        self.current_token = None
        self.token_refresh_task = None
        self.connection_restart_flag = False

        # 消息处理器
        self.handler = MessageHandler(self.myid, context_manager, agent_graph, xianyu_api)
        self.xianyu = xianyu_api
        self.xianyu.session.cookies.update(self.cookies)

    async def refresh_token(self) -> str:
        """刷新 token"""
        try:
            logger.info("开始刷新token...")
            token_result = self.xianyu.get_token(self.device_id)
            if 'data' in token_result and 'accessToken' in token_result['data']:
                self.current_token = token_result['data']['accessToken']
                self.last_token_refresh_time = time.time()
                logger.info("Token刷新成功")
                return self.current_token
            logger.error(f"Token刷新失败: {token_result}")
        except Exception as e:
            logger.error(f"Token刷新异常: {e}")
        return None

    async def token_refresh_loop(self):
        """Token 刷新循环"""
        while True:
            try:
                if time.time() - self.last_token_refresh_time >= self.token_refresh_interval:
                    logger.info("Token即将过期，准备刷新...")
                    if await self.refresh_token():
                        self.connection_restart_flag = True
                        if self.ws:
                            await self.ws.close()
                        break
                    else:
                        await asyncio.sleep(self.token_retry_interval)
                        continue
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Token刷新循环出错: {e}")
                await asyncio.sleep(60)

    async def init(self, ws):
        """初始化连接"""
        if not self.current_token or (time.time() - self.last_token_refresh_time) >= self.token_refresh_interval:
            await self.refresh_token()

        if not self.current_token:
            raise Exception("Token获取失败")

        msg = {
            "lwp": "/reg",
            "headers": {
                "cache-header": "app-key token ua wv",
                "app-key": "444e9908a51d1cb236a27862abc769c9",
                "token": self.current_token,
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36 DingTalk(2.1.5)",
                "dt": "j",
                "wv": "im:3,au:3,sy:6",
                "sync": "0,0;0;0;",
                "did": self.device_id,
                "mid": generate_mid()
            }
        }
        await ws.send(json.dumps(msg))
        await asyncio.sleep(1)

        msg = {
            "lwp": "/r/SyncStatus/ackDiff",
            "headers": {"mid": "5701741704675979 0"},
            "body": [{
                "pipeline": "sync", "tooLong2Tag": "PNM,1", "channel": "sync",
                "topic": "sync", "highPts": 0,
                "pts": int(time.time() * 1000) * 1000,
                "seq": 0, "timestamp": int(time.time() * 1000)
            }]
        }
        await ws.send(json.dumps(msg))
        logger.info('连接注册完成')

    async def main(self):
        """主循环"""
        while True:
            try:
                self.connection_restart_flag = False
                headers = {
                    "Cookie": self.cookies_str,
                    "Host": "wss-goofish.dingtalk.com",
                    "Connection": "Upgrade",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36",
                    "Origin": "https://www.goofish.com",
                }

                async with websockets.connect(self.base_url, extra_headers=headers) as ws:
                    self.ws = ws
                    await self.init(ws)

                    # 启动心跳和 Token 刷新
                    heartbeat_task = self.heartbeat.start(ws)
                    self.token_refresh_task = asyncio.create_task(self.token_refresh_loop())

                    async for message in ws:
                        try:
                            if self.connection_restart_flag:
                                break

                            message_data = json.loads(message)

                            if self.heartbeat.handle_response(message_data):
                                continue

                            await self.handler.handle(message_data, ws)

                        except json.JSONDecodeError:
                            logger.error("消息解析失败")
                        except Exception as e:
                            logger.error(f"处理消息错误: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket连接已关闭")
            except Exception as e:
                logger.error(f"连接错误: {e}")
            finally:
                await self.heartbeat.stop()
                if self.token_refresh_task:
                    self.token_refresh_task.cancel()
                    try:
                        await self.token_refresh_task
                    except asyncio.CancelledError:
                        pass

                if self.connection_restart_flag:
                    logger.info("主动重启连接...")
                else:
                    logger.info("等待5秒后重连...")
                    await asyncio.sleep(5)
