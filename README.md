# Xianyu AutoAgent - 闲鱼智能客服

基于 [shaxiu/XianyuAutoAgent](https://github.com/shaxiu/XianyuAutoAgent) 二次开发，专注于软件开发/程序定制服务场景。

## 主要改动

- 重构为 LangGraph 工作流架构
- 添加 FAISS 向量知识库检索
- 添加情感分析模块
- 精简提示词系统（单一 service_prompt）
- 添加本地测试环境
- 添加飞书通知集成

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填写 API_KEY 和 COOKIES_STR

# 本地测试
python local_chat.py

# 生产运行
python main.py
```

## 配置说明

`.env` 文件：
```
API_KEY=你的LLM API密钥
COOKIES_STR=闲鱼网页端cookies
MODEL_BASE_URL=模型服务地址
MODEL_NAME=模型名称（默认qwen-max）
```

## 项目结构

```
├── main.py              # 生产环境入口
├── local_chat.py        # 本地测试
├── agent/               # Agent核心模块
│   ├── graph.py         # LangGraph工作流
│   ├── knowledge.py     # 知识库RAG
│   └── emotion.py       # 情感分析
├── core/                # 闲鱼平台集成
├── storage/             # 数据存储
├── prompts/             # 提示词
└── knowledge/           # 案例库
```

## 致谢

- 原项目：[shaxiu/XianyuAutoAgent](https://github.com/shaxiu/XianyuAutoAgent)

## 声明

本项目仅供学习交流，请遵守相关平台规则。
