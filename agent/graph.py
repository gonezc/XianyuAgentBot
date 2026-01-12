"""
LangGraph 工作流

整合：State + 节点 + 工作流
支持对话阶段管理
"""
import os
import json
import re
from typing import TypedDict, Annotated, Sequence, Optional, Literal

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger
import sqlite3

# SQLite持久化
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logger.warning("langgraph-checkpoint-sqlite 未安装")

from .tools import tools
from .knowledge import KnowledgeBase
from .emotion import get_emotion_analyzer
from .guardrails import get_guardrails
from .monitor import get_monitor
from .evaluation import get_evaluator


# ============ 对话阶段定义 ============

class Stage:
    """对话阶段"""
    GREETING = "GREETING"           # 开场
    REQUIREMENT = "REQUIREMENT"     # 收集需求
    PRICING = "PRICING"             # 询问期望价格
    NEGOTIATION = "NEGOTIATION"     # 议价
    CLOSING = "CLOSING"             # 成交引导
    COMPLETED = "COMPLETED"         # 已完成


# ============ State ============

class AgentState(TypedDict):
    """Agent状态（节点间共享）"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 对话阶段
    stage: Optional[str]

    # 需求信息
    requirements: Optional[dict]  # {"type": "Web开发", "details": {...}, "missing": [...]}

    # 议价计数
    bargain_count: Optional[int]

    # 价格信息（议价时保持一致）
    quoted_price: Optional[int]  # 已报价格
    floor_price: Optional[int]   # 底价（不能再低）

    # 情感分析
    emotion: Optional[dict]

    # 回复策略
    strategy: Optional[str]

    # 系统提示词（用于调试）
    last_prompt: Optional[str]

    # 其他
    item_desc: Optional[str]
    user_id: Optional[str]
    user_name: Optional[str]


# ============ 需求收集配置 ============

# 不同项目类型需要收集的关键信息
REQUIRED_INFO = {
    "Web开发": ["前端技术", "后端技术", "数据库", "核心功能"],
    "数据分析": ["数据来源", "分析目标", "可视化需求"],
    "爬虫": ["目标网站", "数据量级", "反爬难度"],
    "Python脚本": ["功能需求", "输入输出"],
    "自动化脚本": ["功能需求", "运行环境"],
    "桌面软件": ["技术栈", "核心功能", "平台要求"],
}


# ============ 辅助函数 ============

_kb = None
_llm = None


def _get_kb():
    """获取知识库实例"""
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def _get_llm(temperature: float = 0.7):
    """获取LLM实例"""
    return ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=os.getenv("MODEL_NAME", "qwen-max"),
        temperature=temperature
    )


def _load_prompt():
    """加载提示词"""
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "service_prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "你是一名专业的客服，专注于软件开发和程序定制服务。"


def _load_knowledge():
    """预加载知识"""
    kb = _get_kb()
    ctx = ""
    if kb.skills:
        ctx += f"\n【技术能力】\n{kb.skills}\n"
    return ctx


def _format_messages(messages: Sequence[BaseMessage], limit: int = 10) -> str:
    """格式化消息历史"""
    recent = list(messages)[-limit:]
    lines = []
    for msg in recent:
        role = "用户" if isinstance(msg, HumanMessage) else "客服"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if content:  # 跳过空消息（如工具调用）
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_json_response(text: str) -> dict:
    """从LLM响应中提取JSON"""
    # 尝试找到 JSON 块
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ============ 节点1：上下文分析 ============

ANALYSIS_PROMPT = """分析以下对话，判断当前状态。

## 对话历史
{conversation}

## 当前用户消息
{last_message}

## 请返回JSON格式（严格按此格式）：
```json
{{
    "stage": "阶段，从以下选择：GREETING|REQUIREMENT|PRICING|NEGOTIATION|CLOSING",
    "emotion": {{
        "sentiment": "positive|negative|neutral"
    }},
    "requirements": {{
        "type": "项目类型，如：Web开发/爬虫/数据分析/Python脚本/自动化脚本/桌面软件，未知填null",
        "project_name": "用户原始描述的项目名称，如'炒股脚本'、'图书管理系统'，必须保留用户原话！",
        "details": {{
            "features": "用户提到的具体功能，必须保留用户原话！如用户说'自动买卖、日志记录'就填这个",
            "tech_stack": "技术栈要求，如'Spring Boot+Vue'，未提及填null",
            "deadline": "截止时间，未提及填null"
        }},
        "expected_price": "用户期望价格，未提及填null",
        "expected_time": "用户期望工期，未提及填null",
        "missing": ["还需要了解的信息"]
    }},
    "quoted_price": "客服已报价格（数字），如客服说'最少3200'则填3200，未报价填null",
    "floor_price": "客服提到的底价（数字），未提及填null",
    "is_bargaining": true或false,
    "ready_to_close": true或false
}}
```

## 阶段判断规则（重要！）：
1. GREETING：首次对话或打招呼
2. REQUIREMENT：正在了解需求，【项目类型、技术栈、功能三者缺一不可，否则都是这个阶段】
3. PRICING：需求基本明确，且用户说了预期价格/预算
4. NEGOTIATION：已报价，用户在讨价还价（说"贵了"、给出期望价格等）
5. CLOSING：用户表示同意/确认成交（说"行"、"可以"、"OK"、"好"、"就这样"、"xxx吧"、"成交"等）

## 注意：
- 如果只知道"图书管理系统"但不知道技术栈或功能，stage应该是REQUIREMENT
- 【重要】用户说了预期价格（如"预算50"、"500块"），且需求基本明确，就进入PRICING阶段
- 用户同意卖家报价时（如"1200吧"、"行吧"、"可以"），应该进入CLOSING阶段
- 【重要】提取客服已报价格时，找客服说的"最少xxx"、"最低xxx"、"xxx起"等表述中的数字

请直接返回JSON，不要其他解释。"""


def analyze_context(state: AgentState) -> dict:
    """节点1：分析对话上下文（用LLM）"""
    messages = state.get("messages", [])
    if not messages:
        return {
            "stage": Stage.GREETING,
            "emotion": {"sentiment": "neutral"},
            "requirements": {"type": None, "details": {}, "missing": ["项目类型"]},
            "bargain_count": 0
        }
    
    # 获取消息历史
    last_msg = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
    conversation = _format_messages(messages[:-1]) if len(messages) > 1 else ""
    
    # 使用情绪分析模型分析情绪
    emotion_result = None
    try:
        emotion_analyzer = get_emotion_analyzer()
        emotion_result = emotion_analyzer.analyze(last_msg, context=conversation)
        logger.info(f"情绪分析结果: {emotion_result}")
    except Exception as e:
        logger.warning(f"情绪分析失败: {e}，将使用LLM分析的情绪结果")
    
    # 首条消息直接判断为开场/需求阶段
    if len(messages) == 1:
        # 简单关键词判断是否直接问需求
        if any(kw in last_msg for kw in ["多少钱", "报价", "价格", "能做", "可以做"]):
            stage = Stage.REQUIREMENT
        else:
            stage = Stage.GREETING
        
        # 使用情绪分析结果，如果没有则使用默认值
        emotion = emotion_result if emotion_result else {
            "sentiment": "neutral"
        }
        
        return {
            "stage": stage,
            "emotion": emotion,
            "requirements": {"type": None, "details": {}, "missing": ["项目类型", "具体需求"]},
            "bargain_count": 0
        }
    
    # 用LLM分析阶段和需求
    prompt = ANALYSIS_PROMPT.format(
        conversation=conversation,
        last_message=last_msg
    )
    
    try:
        llm = _get_llm(temperature=0.3)  # 低温度，更稳定
        response = llm.invoke([HumanMessage(content=prompt)])
        result = _parse_json_response(response.content)
        
        if result:
            # 更新议价计数
            bargain_count = state.get("bargain_count", 0)
            if result.get("is_bargaining"):
                bargain_count += 1
            
            # 优先使用模型分析的情绪结果
            if emotion_result:
                final_emotion = emotion_result.copy()
            else:
                final_emotion = result.get("emotion", {"sentiment": "neutral"})

            logger.info(f"上下文分析: stage={result.get('stage')}, emotion={final_emotion}")
            logger.info(f"需求信息: {result.get('requirements', {})}")

            # 提取价格信息（优先使用LLM提取的，否则保留之前的）
            quoted_price = result.get("quoted_price") or state.get("quoted_price")
            floor_price = result.get("floor_price") or state.get("floor_price")

            return {
                "stage": result.get("stage", Stage.REQUIREMENT),
                "emotion": final_emotion,
                "requirements": result.get("requirements", {}),
                "bargain_count": bargain_count,
                "quoted_price": quoted_price,
                "floor_price": floor_price
            }
    except Exception as e:
        logger.error(f"上下文分析失败: {e}")
    
    # 降级：使用简单规则
    return {
        "stage": state.get("stage", Stage.REQUIREMENT),
        "emotion": {"sentiment": "neutral"},
        "requirements": state.get("requirements", {}),
        "bargain_count": state.get("bargain_count", 0),
        "quoted_price": state.get("quoted_price"),
        "floor_price": state.get("floor_price")
    }


# ============ 节点2：策略选择 ============

# 策略模板（简洁明确）
STRATEGY_TEMPLATES = {
    # GREETING阶段
    "greeting": "回复：想做啥项目？有啥需求可以说说",

    # REQUIREMENT阶段（不绑定工具，无法调用）
    "ask_type": "追问项目类型，不报价",
    "ask_tech": "追问：用什么技术栈？前后端分别是啥？",
    "ask_features": "追问：主要需要哪些功能？",
    "ask_budget": "追问：大概预算多少？啥时候要？",

    # PRICING阶段 - 核心决策
    "search_and_decide": """先调用 search_cases(query='{query}') 搜索案例，等待返回结果后评估底价：

【底价计算】（found=true时，必须严格按此计算！）
- 底价 = price_range[0]（返回结果中的最低价，不打折！）
- 例如：price_range=[2000, 4000]，底价就是2000
- 功能更多可上浮10-30%，工期紧（<5天）可上浮20%

【决策】
- 预期 >= 底价: 回复"可以，下单我改价"
- 预期 < 底价: 回复"有点低，这个最少xxx"（给出底价，不要编数字！）
- found=false: 调用 send_reminder(notice_type='handover')，回复"这个比较特殊，加微信详聊"

【重要】底价必须基于 price_range[0]，禁止编造数字！""",

    # NEGOTIATION阶段（必须基于已报价格！）
    "bargain_1": "首次议价：之前报价{quoted_price}，底价{floor_price}，可优惠5-10%。强调包含完整源码+部署文档+7天售后",
    "bargain_2": "二次议价：之前报价{quoted_price}，底价{floor_price}，接近底价了。强调代码规范、有注释、方便后期维护",
    "bargain_3": "三次议价：坚持底价{floor_price}，强调交付质量：源码+文档+演示+售后，比淘宝便宜货靠谱",
    "bargain_final": "多次议价+情绪差：给最终底价{floor_price}，强调一分钱一分货，或建议缩减功能降低成本",

    # CLOSING阶段
    "closing": """引导下单：
1. 让客户点"我想要"下单
2. 告知会改价到约定金额
3. 给微信号详聊需求
4. 调用 send_reminder(message='{summary}', notice_type='reminder')"""
}


def select_strategy(state: AgentState) -> dict:
    """节点2：基于阶段选择回复策略"""
    stage = state.get("stage", Stage.GREETING)
    emotion = state.get("emotion", {})
    requirements = state.get("requirements", {})
    bargain_count = state.get("bargain_count", 0)

    project_type = requirements.get("type", "")
    details = requirements.get("details", {})
    features = details.get("features") or details.get("功能") or ""
    tech_stack = details.get("tech_stack") or details.get("技术栈") or ""
    expected_price = requirements.get("expected_price", "")

    # 根据阶段选择策略
    # 【保险逻辑】如果用户已经给了预期价格，强制进入 PRICING 流程
    if expected_price and stage == Stage.REQUIREMENT:
        stage = Stage.PRICING
        logger.info(f"用户已给预期价格 {expected_price}，强制进入 PRICING 阶段")

    if stage == Stage.GREETING:
        strategy = STRATEGY_TEMPLATES["greeting"]

    elif stage == Stage.REQUIREMENT:
        if not project_type:
            strategy = STRATEGY_TEMPLATES["ask_type"]
        elif not tech_stack:
            strategy = STRATEGY_TEMPLATES["ask_tech"]
        elif not features:
            strategy = STRATEGY_TEMPLATES["ask_features"]
        else:
            strategy = STRATEGY_TEMPLATES["ask_budget"]

    elif stage == Stage.PRICING:
        if not expected_price:
            strategy = STRATEGY_TEMPLATES["ask_budget"]
        else:
            # 构建搜索关键词：像人话一样描述项目
            project_name = requirements.get("project_name", "") or project_type
            if tech_stack and features:
                query = f"基于{tech_stack}的{project_name}，包含{features}等功能"
            elif tech_stack:
                query = f"基于{tech_stack}的{project_name}"
            elif features:
                query = f"{project_name}，包含{features}等功能"
            else:
                query = project_name
            strategy = STRATEGY_TEMPLATES["search_and_decide"].format(
                query=query,
                price=expected_price,
                min_price="案例最低价",
                suggest_price="案例价"
            )

    elif stage == Stage.NEGOTIATION:
        sentiment = emotion.get("sentiment", "neutral")
        # 获取已报价格和底价
        quoted_price = state.get("quoted_price", "未知")
        floor_price = state.get("floor_price", "未知")

        if bargain_count >= 3 and sentiment == "negative":
            strategy = STRATEGY_TEMPLATES["bargain_final"].format(
                quoted_price=quoted_price, floor_price=floor_price
            )
        elif bargain_count >= 3:
            strategy = STRATEGY_TEMPLATES["bargain_3"].format(
                quoted_price=quoted_price, floor_price=floor_price
            )
        elif bargain_count == 2:
            strategy = STRATEGY_TEMPLATES["bargain_2"].format(
                quoted_price=quoted_price, floor_price=floor_price
            )
        else:
            strategy = STRATEGY_TEMPLATES["bargain_1"].format(
                quoted_price=quoted_price, floor_price=floor_price
            )

    elif stage == Stage.CLOSING:
        # 构建详细的成交通知
        project_name = requirements.get("project_name", "") or project_type
        expected_time = requirements.get("expected_time", "") or details.get("deadline", "")
        quoted_price = state.get("quoted_price", "")
        user_name = state.get("user_name", "")
        item_desc = state.get("item_desc", "")

        summary = f"【即将成交】\n用户: {user_name}\n商品: {item_desc}\n项目: {project_name}\n技术栈: {tech_stack}\n功能: {features}\n价格: {quoted_price or expected_price}\n工期: {expected_time}"
        strategy = STRATEGY_TEMPLATES["closing"].format(summary=summary)

    else:
        strategy = "正常专业解答"

    logger.info(f"策略: [{stage}] {strategy[:60]}...")
    return {"strategy": strategy}


# ============ 节点3：调用LLM ============

def call_model(state: AgentState) -> dict:
    """节点3：调用LLM生成回复"""
    stage = state.get("stage", Stage.GREETING)
    strategy = state.get("strategy", "正常解答")
    requirements = state.get("requirements", {})
    monitor = get_monitor()

    # 构建系统提示
    # 如果已有期望时间，就不在details中显示deadline
    details = requirements.get('details', {}).copy() if requirements.get('details') else {}
    if requirements.get('expected_time'):
        details.pop('deadline', None)  # 移除deadline字段
    details_str = json.dumps(details, ensure_ascii=False) if details else '暂无'

    system_content = f"""{_load_prompt()}

{_load_knowledge()}

【当前状态】
- 对话阶段：{stage}
- 商品信息：{state.get('item_desc', '暂无')}
- 已知需求：{details_str}
- 期望价格：{requirements.get('expected_price', '未知')}
- 期望时间：{requirements.get('expected_time', '未知')}
- 已报价格：{state.get('quoted_price', '未报价')}
- 底价：{state.get('floor_price', '未定')}

【本轮策略 - 必须严格执行！】
{strategy}

注意：你必须按照上面的策略回复，不要跳过步骤！议价时必须基于已报价格，不能随意编造新价格！"""

    messages = [SystemMessage(content=system_content)] + list(state["messages"])

    try:
        llm = _get_llm()
        # REQUIREMENT阶段不绑定工具，禁止调用
        if stage in [Stage.GREETING, Stage.REQUIREMENT]:
            response = llm.invoke(messages)
        else:
            response = llm.bind_tools(tools).invoke(messages)

        # 记录 Token 用量
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            monitor.record_tokens(
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )

        # 记录工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                monitor.record_tool_call(tc.get('name', ''))

        content = response.content if response.content else "[工具调用]"
        logger.info(f"LLM: {content[:100]}...")

        # 保存系统提示词到state，方便调试查看
        return {"messages": [response], "last_prompt": system_content}

    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        monitor.end_call(success=False, error=str(e))

        # Fallback: 返回兜底回复
        fallback_response = monitor.get_fallback_response()
        return {"messages": [AIMessage(content=fallback_response)], "last_prompt": system_content}






# ============ 路由函数 ============

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """路由：是否需要调用工具"""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            logger.info(f"调用工具: {[tc['name'] for tc in last.tool_calls]}")
            return "tools"
    return "__end__"


def should_generate_reply_after_tools(state: AgentState) -> Literal["call_model", "__end__"]:
    """路由：工具执行后是否需要生成回复

    以下情况不再生成回复，直接结束：
    1. 发送了转人工通知（send_reminder with notice_type='handover'）
    2. 之前的 AIMessage 已经有内容（不只是工具调用）
    """
    messages = state.get("messages", [])

    # 从后往前查找最近的 AIMessage（包含工具调用）
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # 检查这个 AIMessage 是否已经有内容
            if msg.content and msg.content.strip() and msg.content != "[工具调用]":
                logger.info(f"AIMessage 已有内容，不再生成新回复: {msg.content[:50]}...")
                return "__end__"

            # 检查是否是转人工通知
            for tc in msg.tool_calls:
                if tc.get('name') == 'send_reminder' and tc.get('args', {}).get('notice_type') == 'handover':
                    # 找到对应的 ToolMessage 检查是否成功
                    for tool_msg in reversed(messages):
                        if isinstance(tool_msg, ToolMessage) and hasattr(tool_msg, 'tool_call_id'):
                            if tc.get('id') == getattr(tool_msg, 'tool_call_id', None):
                                try:
                                    content = tool_msg.content if hasattr(tool_msg, 'content') else str(tool_msg)
                                    result = json.loads(content) if isinstance(content, str) else content
                                    if result.get("success", False):
                                        logger.info("检测到转人工通知已发送，不再生成回复")
                                        return "__end__"
                                except Exception as e:
                                    logger.warning(f"解析工具结果失败: {e}")
            break

    # 需要生成回复
    return "call_model"


# ============ 工作流构建 ============

_db_conn = None


def _get_db_connection(db_path: str):
    """获取数据库连接"""
    global _db_conn
    if _db_conn is None:
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        _db_conn = sqlite3.connect(db_path, check_same_thread=False)
    return _db_conn


def create_workflow(memory_type: str = "sqlite", db_path: str = "data/chat_history.db"):
    """创建工作流
    
    Args:
        memory_type: "sqlite"(持久化) | "memory"(内存) | "none"(无记忆)
        db_path: 数据库路径
    """
    logger.info("创建LangGraph工作流...")
    
    workflow = StateGraph(AgentState)
    
    # 设置结点
    workflow.add_node("analyze_context", analyze_context)
    workflow.add_node("select_strategy", select_strategy)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", ToolNode(tools))  # 执行工具
    
    # 设置边
    workflow.set_entry_point("analyze_context")
    workflow.add_edge("analyze_context", "select_strategy")
    workflow.add_edge("select_strategy", "call_model")
    workflow.add_conditional_edges("call_model", should_continue, {
        "tools": "tools",
        "__end__": END
    })
    # 工具执行后，检查是否发送了转人工通知，决定是否生成回复
    workflow.add_conditional_edges("tools", should_generate_reply_after_tools, {
        "call_model": "call_model",  # 需要生成回复
        "__end__": END  # 转人工通知已发送，不生成回复，直接结束
    })
    
    # 编译
    if memory_type == "sqlite" and SQLITE_AVAILABLE:
        conn = _get_db_connection(db_path)
        graph = workflow.compile(checkpointer=SqliteSaver(conn))
        logger.info(f"工作流创建完成（SQLite: {db_path}）")
    elif memory_type == "memory":
        graph = workflow.compile(checkpointer=MemorySaver())
        logger.info("工作流创建完成（内存）")
    else:
        graph = workflow.compile()
        logger.info("工作流创建完成（无记忆）")
    
    return graph


def process_message(graph, user_msg: str, item_desc: str = "",
                    user_id: str = "", user_name: str = "",
                    thread_id: str = "default",
                    db_path: str = "data/chat_history.db",
                    return_state: bool = False):
    """处理用户消息

    Args:
        graph: 工作流实例
        user_msg: 用户消息
        item_desc: 商品描述
        thread_id: 会话ID
        db_path: 数据库路径
        return_state: 是否返回完整状态（用于调试）

    Returns:
        str: AI回复
        或 (str, dict): (AI回复, 完整状态) 如果 return_state=True
    """
    from .storage import get_store
    store = get_store(db_path)
    guardrails = get_guardrails()
    monitor = get_monitor()
    evaluator = get_evaluator()

    # 检查该 thread_id 是否处于转人工状态
    if store.is_handover(thread_id):
        logger.info(f"thread_id={thread_id} 处于转人工状态，跳过AI回复")
        # 保存用户消息但不回复
        store.save_message(thread_id, "user", user_msg, item_desc)
        if return_state:
            return "", {"messages": [], "handover": True}
        return ""

    # Guardrails: 检查用户输入
    _, should_respond = guardrails.process_input(user_msg)
    if not should_respond:
        logger.warning(f"用户输入被安全规则拦截: {user_msg[:50]}...")
        store.save_message(thread_id, "user", user_msg, item_desc)
        if return_state:
            return "", {"messages": [], "blocked": True}
        return ""

    config = {"configurable": {"thread_id": thread_id}}

    # 保存用户消息
    store.save_message(thread_id, "user", user_msg, item_desc)

    # Monitor: 开始记录
    monitor.start_call(thread_id)

    result = graph.invoke({
        "messages": [HumanMessage(content=user_msg)],
        "item_desc": item_desc,
        "user_id": user_id,
        "user_name": user_name
    }, config=config)

    # Evaluator: 更新对话统计
    evaluator.update_conversation(
        thread_id,
        stage=result.get("stage"),
        bargain_count=result.get("bargain_count")
    )

    messages = result.get("messages", [])

    # 提取AI回复（如果存在）
    reply = ""
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            reply = last_msg.content if hasattr(last_msg, 'content') and last_msg.content else ""

    # 检查是否发送了转人工通知
    is_handover = False
    if not reply:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get('name') == 'send_reminder' and tc.get('args', {}).get('notice_type') == 'handover':
                        # 检查工具执行结果
                        for tool_msg in messages:
                            if isinstance(tool_msg, ToolMessage) and hasattr(tool_msg, 'tool_call_id'):
                                if tc.get('id') == getattr(tool_msg, 'tool_call_id', None):
                                    try:
                                        content = tool_msg.content if hasattr(tool_msg, 'content') else str(tool_msg)
                                        result_data = json.loads(content) if isinstance(content, str) else content
                                        if result_data.get("success", False):
                                            is_handover = True
                                            # 标记该 thread_id 为转人工状态
                                            store.set_handover(thread_id, is_handover=True)
                                            logger.info(f"转人工通知已发送，标记 thread_id={thread_id} 为转人工状态")
                                    except Exception as e:
                                        logger.warning(f"检查转人工通知失败: {e}")
                break

    # Guardrails: 处理AI输出
    if reply:
        reply = guardrails.process_output(reply)

    # Monitor: 结束记录
    monitor.end_call(success=True)

    # 如果有回复才保存到数据库
    if reply:
        store.save_message(
            thread_id, "assistant", reply, item_desc,
            emotion=result.get("emotion", {}),
            strategy=result.get("strategy", ""),
            stage=result.get("stage", "")
        )
    elif is_handover:
        logger.info(f"转人工通知已发送（thread_id={thread_id}），未保存回复到数据库")

    if return_state:
        return reply, result
    return reply
