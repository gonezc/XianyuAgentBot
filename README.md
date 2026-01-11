# XianyuServiceBot

闲鱼智能客服系统，基于 LangGraph 工作流的 AI 自动值守方案，专注于软件开发/程序定制服务场景。

## 核心特性

- 多阶段对话管理：基于 LangGraph 状态机实现开场、需求收集、询价、议价、成交引导的完整销售流程
- 知识库检索：FAISS 向量检索支持案例库匹配，提供智能定价参考
- 情感分析：实时分析买家情绪，动态调整回复策略
- 对话持久化：SQLite 存储完整对话历史，支持上下文感知
- 飞书通知：订单确认时自动发送飞书卡片通知
- 人工接管：支持关键词切换人工/AI 模式
- 本地调试：提供独立测试环境，无需连接闲鱼平台即可调试

## 技术框架

| 模块 | 技术栈 |
|------|--------|
| LLM | OpenAI 接口规范 |
| 工作流 | LangGraph |
| 向量检索 | FAISS |
| 情感分析 | Erlangshen-Roberta-110M-Sentiment (BERT) |
| 数据存储 | SQLite |
| 平台通信 | WebSocket |

项目结构：

```
├── main.py              # 生产环境入口
├── local_chat.py        # 本地测试环境
├── agent/               # Agent 核心模块
│   ├── graph.py         # LangGraph 工作流
│   ├── knowledge.py     # 知识库 RAG
│   ├── emotion.py       # 情感分析
│   └── tools.py         # LLM 工具
├── core/                # 闲鱼平台集成
│   ├── websocket_client.py
│   └── message_handler.py
├── storage/             # 数据存储
├── prompts/             # 提示词配置
├── knowledge/           # 案例库
└── images/              # 效果图（不开源）
```

## 快速运行

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

创建 `.env` 文件：

```
API_KEY=你的LLM API密钥
COOKIES_STR=闲鱼网页端cookies
MODEL_BASE_URL=模型服务地址
MODEL_NAME=模型名称
```

COOKIES_STR 获取方式：闲鱼网页端 F12 打开控制台，Network - Fetch/XHR，复制任意请求的 Cookie。

3. 运行

```bash
# 本地测试
python local_chat.py

# 生产运行
python main.py
```

## 效果图

<!-- 效果图待补充 -->

## 致谢

- 原项目：[shaxiu/XianyuAutoAgent](https://github.com/shaxiu/XianyuAutoAgent)
- [changan593](https://github.com/changan593)

## 声明

本项目仅供学习交流，请遵守相关平台规则。
