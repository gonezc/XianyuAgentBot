"""
本地对话测试环境

模拟闲鱼对话，无需连接闲鱼平台
"""
import os
import sys
from dotenv import load_dotenv

# Windows终端编码处理
# 确保中文输入输出正常
if sys.platform == 'win32':
    import locale
    
    # 设置控制台代码页为UTF-8（如果可能）
    try:
        os.system('chcp 65001 > nul 2>&1')
    except:
        pass
    
    # 注意：不要重新包装 stdin，否则 input() 可能无法正常工作
    # Python 会根据终端编码自动处理，Windows CMD 默认 GBK，PowerShell 可能支持 UTF-8

# 加载环境变量
load_dotenv()

def print_header():
    print("\n" + "=" * 60)
    print("  闲鱼AI客服 - 本地测试环境")
    print("=" * 60)
    print("命令：")
    print("  /quit     - 退出")
    print("  /clear    - 清空对话（新建会话）")
    print("  /order    - 模拟买家下单付款（触发飞书通知）")
    print("  /emotion  - 查看上次情感分析结果")
    print("  /strategy - 查看上次策略")
    print("  /prompt   - 查看上一轮的系统提示词")
    print("  /tools    - 查看上一轮的工具调用结果")
    print("  /messages - 查看LLM看到的所有消息（完整对话历史）")
    print("  /item     - 设置商品描述")
    print("  /debug    - 开关调试模式")
    print("  /history  - 查看当前会话ID")
    print("-" * 60)
    print("对话历史自动保存到 data/chat_history.db")
    print("=" * 60 + "\n")


def main():
    # 检查API_KEY
    if not os.getenv("API_KEY"):
        print("错误: API_KEY未配置，请在.env文件中设置")
        sys.exit(1)
    
    from agent import create_workflow, process_message
    from langchain_core.messages import HumanMessage, AIMessage
    
    print_header()
    
    # 创建工作流（使用SQLite持久化）
    print("正在初始化Agent...")
    graph = create_workflow(memory_type="sqlite", db_path="data/chat_history.db")
    print("初始化完成!\n")
    
    # 状态（使用时间戳作为会话ID，方便区分不同测试）
    import time
    thread_id = f"local_{int(time.time())}"
    item_desc = "Python代做/软件开发/程序定制"
    debug_mode = False
    last_emotion = {}
    last_strategy = ""
    last_prompt = ""
    last_tool_results = []
    last_messages = []
    
    print(f"当前商品: {item_desc}")
    print("开始对话 (输入 /quit 退出)\n")
    
    while True:
        try:
            user_input = input("买家> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break
        except Exception as e:
            print(f"\n[错误] 输入错误: {e}\n")
            continue
        
        if not user_input:
            continue
        
        # 处理命令
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd == "/quit":
                print("再见!")
                break
            
            elif cmd == "/clear":
                thread_id = f"local_test_{id(object())}"  # 新会话ID
                print("[系统] 对话历史已清空\n")
                continue
            
            elif cmd == "/emotion":
                print(f"[情感分析] {last_emotion}\n")
                continue
            
            elif cmd == "/strategy":
                print(f"[策略] {last_strategy}\n")
                continue
            
            elif cmd == "/prompt":
                if last_prompt:
                    print("\n" + "=" * 60)
                    print("上一轮系统提示词:")
                    print("=" * 60)
                    print(last_prompt)
                    print("=" * 60 + "\n")
                else:
                    print("[系统] 暂无系统提示词（请先发送一条消息）\n")
                continue
            
            elif cmd == "/tools":
                if last_tool_results:
                    print("\n" + "=" * 60)
                    print("上一轮工具调用结果:")
                    print("=" * 60)
                    for i, tool_result in enumerate(last_tool_results, 1):
                        tool_name = tool_result.get("name", "未知工具")
                        tool_content = tool_result.get("content", "")
                        print(f"\n[{i}] 工具: {tool_name}")
                        print("-" * 60)
                        # 尝试格式化JSON
                        try:
                            import json
                            parsed = json.loads(tool_content)
                            print(json.dumps(parsed, ensure_ascii=False, indent=2))
                        except:
                            print(tool_content)
                    print("=" * 60 + "\n")
                else:
                    print("[系统] 暂无工具调用结果（上一轮未调用工具）\n")
                continue
            
            elif cmd == "/messages":
                if last_messages:
                    print("\n" + "=" * 60)
                    print("LLM看到的所有消息（完整对话历史）:")
                    print("=" * 60)
                    for i, msg in enumerate(last_messages, 1):
                        msg_type = type(msg).__name__
                        print(f"\n[{i}] {msg_type}")
                        print("-" * 60)
                        
                        # SystemMessage
                        if msg_type == "SystemMessage":
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            print(content[:500] + "..." if len(content) > 500 else content)
                        
                        # HumanMessage
                        elif msg_type == "HumanMessage":
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            print(f"用户: {content}")
                        
                        # AIMessage
                        elif msg_type == "AIMessage":
                            content = msg.content if hasattr(msg, 'content') else "[工具调用]"
                            print(f"AI: {content}")
                            # 如果有tool_calls，显示
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                print(f"\n工具调用:")
                                for tc in msg.tool_calls:
                                    print(f"  - {tc.get('name', 'unknown')}({tc.get('args', {})})")
                        
                        # ToolMessage
                        elif msg_type == "ToolMessage":
                            tool_name = getattr(msg, 'name', 'unknown')
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            print(f"工具: {tool_name}")
                            # 尝试格式化JSON
                            try:
                                import json
                                parsed = json.loads(content)
                                print(json.dumps(parsed, ensure_ascii=False, indent=2))
                            except:
                                print(content[:500] + "..." if len(content) > 500 else content)
                        
                        else:
                            print(str(msg)[:500])
                    
                    print("=" * 60 + "\n")
                else:
                    print("[系统] 暂无消息历史（请先发送一条消息）\n")
                continue
            
            elif cmd.startswith("/item"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    item_desc = parts[1]
                    print(f"[系统] 商品描述已更新: {item_desc}\n")
                else:
                    print(f"[当前商品] {item_desc}")
                    print("用法: /item 新的商品描述\n")
                continue
            
            elif cmd == "/debug":
                debug_mode = not debug_mode
                print(f"[系统] 调试模式: {'开启' if debug_mode else '关闭'}\n")
                continue
            
            elif cmd == "/history":
                print(f"[系统] 当前会话ID: {thread_id}")
                print(f"[系统] 数据库: data/chat_history.db\n")
                continue
            
            elif cmd.startswith("/order"):
                # 模拟下单付款
                from agent.notify import send_feishu
                
                # 解析参数: /order 需求 价格 工期
                parts = user_input.split(maxsplit=3)
                if len(parts) >= 4:
                    _, req, price, deadline = parts
                else:
                    req = last_strategy if last_strategy else "测试订单"
                    price = "1000元"
                    deadline = "7天"
                
                print(f"\n[系统] 模拟下单付款...")
                print(f"  需求: {req}")
                print(f"  价格: {price}")
                print(f"  工期: {deadline}")
                
                result = send_feishu(
                    requirement=req,
                    price=price,
                    deadline=deadline,
                    buyer_name="测试买家",
                    buyer_id="test_user"
                )
                
                if result.get("success"):
                    print(f"  [OK] 飞书通知已发送\n")
                else:
                    print(f"  [!] {result.get('message')}\n")
                continue
            
            else:
                print(f"[系统] 未知命令: {cmd}\n")
                continue
        
        # 调用Agent（使用process_message，会同时写入数据库）
        try:
            response, result = process_message(
                graph=graph,
                db_path="data/chat_history.db",
                user_msg=user_input,
                item_desc=item_desc,
                thread_id=thread_id,
                user_id="test_user",
                user_name="测试买家",
                return_state=True  # 返回完整状态用于调试
            )
            
            # 保存状态用于调试
            last_emotion = result.get("emotion", {}) if result else {}
            last_strategy = result.get("strategy", "") if result else ""
            last_stage = result.get("stage", "") if result else ""
            last_prompt = result.get("last_prompt", "") if result else ""
            
            # 保存完整的messages历史（LLM看到的所有内容）
            last_messages = result.get("messages", []) if result else []
            
            # 提取工具调用结果（从messages中查找ToolMessage）
            last_tool_results = []
            if result:
                messages = result.get("messages", [])
                from langchain_core.messages import ToolMessage
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        last_tool_results.append({
                            "name": getattr(msg, 'name', 'unknown'),
                            "content": msg.content if hasattr(msg, 'content') else str(msg)
                        })
            
            # 调试信息
            if debug_mode:
                print(f"\n[DEBUG] 阶段: {last_stage}")
                print(f"[DEBUG] 情感: {last_emotion}")
                print(f"[DEBUG] 策略: {last_strategy}")
            
            print(f"\n客服> {response}\n")
            
        except Exception as e:
            print(f"\n[错误] {e}\n")


if __name__ == "__main__":
    main()

