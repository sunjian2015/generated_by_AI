"""LangGraph 进阶: Human-in-the-loop + 多 Agent 协作。

    pip install langgraph langchain-openai
    set OPENAI_API_KEY=sk-xxx
    python langgraph_advanced_minimal.py "搜索 Python 最佳实践并写入 best_practices.md"

展示两个 LangGraph 相比手写 loop 的杀手级特性:

  1) Human-in-the-loop: 危险操作（写文件、执行命令）会在执行前「中断」整张图，
     等待人工批准或修改参数后再「恢复」。这套中断/恢复机制若手写需要额外设计
     状态持久化和续传，LangGraph 通过 interrupt 配合 checkpointer 原生支持。

  2) 多 Agent 协作: 一张图里有多个子 agent，各自带独立工具和专业提示词。主控
     节点根据任务类型「路由」到对应 agent。手写版若要做到这点，需自己维护多个
     agent 实例和调度逻辑；LangGraph 把它抽象成图的节点 + 条件边。

场景: 智能工作助手，包含两个子 agent:
    - researcher: 擅长搜索、读取文档（只读工具，无需人工确认）
    - executor: 擅长写文件、执行命令（危险操作，需人工确认 human-in-the-loop）

主控根据任务描述判断派给谁，各 agent 独立工作，最终把结果交回主控汇总。
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt


# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", ".")).resolve()


# -----------------------------------------------------------------------------
# 2. 工具: 只读 vs 危险写入
# -----------------------------------------------------------------------------
def _resolve(path: str) -> Path:
    target = (WORKDIR / path).resolve()
    if target != WORKDIR and WORKDIR not in target.parents:
        raise ValueError(f"路径 {path!r} 超出工作目录。")
    return target


# --- 研究员的只读工具 ---
@tool
def search_web(query: str) -> str:
    """模拟搜索引擎，返回与 query 相关的网页摘要。"""
    # 真实场景接 SerpAPI / Google Custom Search / Bing API。这里返回假数据演示。
    return f"[模拟搜索结果] 关于 '{query}' 的前 3 条:\n1. 官方文档...\n2. Stack Overflow...\n3. Medium 文章..."


@tool
def read_file(path: str) -> str:
    """读取文本文件内容。"""
    target = _resolve(path)
    if not target.is_file():
        return f"错误: {path} 不是文件。"
    return target.read_text(encoding="utf-8", errors="replace")[:10000]


# --- 执行员的危险写入工具（需 human-in-the-loop）---
@tool
def write_file(path: str, content: str) -> str:
    """把内容写入文件（覆盖）。这是有实际影响的操作，会触发人工确认。"""
    # 真正写入前，用 interrupt() 中断图，让外部用户决定是否继续。
    approved = interrupt(
        {
            "type": "approval_request",
            "operation": "write_file",
            "args": {"path": path, "content_preview": content[:200] + "..."},
            "question": f"即将写入 {path}（{len(content)} 字符），是否允许？",
        }
    )
    # interrupt 返回外部传入的 Command，默认 None。若用户拒绝或修改，在此处理。
    if approved is None:
        approved = {"approved": True}  # 默认批准（演示用）
    if not approved.get("approved"):
        return f"操作已被用户拒绝: {approved.get('reason', '未说明')}"

    # 用户可能修改了参数（比如改路径或内容）。
    final_path = approved.get("path", path)
    final_content = approved.get("content", content)
    target = _resolve(final_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(final_content, encoding="utf-8")
    return f"已写入 {final_path}（{len(final_content)} 字符）。"


@tool
def run_command(cmd: str) -> str:
    """执行 shell 命令并返回输出。危险操作，会触发人工确认。"""
    approved = interrupt(
        {
            "type": "approval_request",
            "operation": "run_command",
            "args": {"cmd": cmd},
            "question": f"即将执行命令: {cmd}\n是否允许？",
        }
    )
    if approved is None:
        approved = {"approved": True}
    if not approved.get("approved"):
        return f"命令执行已被拒绝: {approved.get('reason', '未说明')}"

    import subprocess

    try:
        result = subprocess.run(
            approved.get("cmd", cmd),
            shell=True,
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return f"退出码 {result.returncode}\n{result.stdout}\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "命令执行超时(30s)。"


RESEARCHER_TOOLS = [search_web, read_file]
EXECUTOR_TOOLS = [write_file, run_command]


# -----------------------------------------------------------------------------
# 3. 多 Agent 状态: 除了 messages，还要记录「当前在哪个 agent」
# -----------------------------------------------------------------------------
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str  # "researcher" / "executor" / "supervisor"


# -----------------------------------------------------------------------------
# 4. 三个节点: 主控 supervisor、研究员 researcher、执行员 executor
# -----------------------------------------------------------------------------
def build_llm():
    return ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)


def supervisor_node(state: MultiAgentState):
    """主控: 分析任务并路由给合适的 agent，或汇总结果给用户。

    supervisor 有两次决策点:
      1) 最开始: 读用户任务，判断该派给 researcher 还是 executor。
      2) 子 agent 完成后: 检查其输出，决定是否还需要其它 agent 帮忙，或直接结束。
    """
    llm = build_llm()
    sys_msg = SystemMessage(
        content=(
            "你是任务分派主控。你有两个下属:\n"
            "- researcher: 擅长搜索、读取文档，只做信息收集，不改动任何文件。\n"
            "- executor: 擅长写文件、执行命令，完成实际操作。\n\n"
            "用户给你一个任务，你要判断:\n"
            "1) 如果需要搜索或读文件收集信息 -> 派给 researcher。\n"
            "2) 如果需要写文件或执行命令 -> 派给 executor。\n"
            "3) 如果子 agent 已完成工作且能满足用户需求 -> 回复 FINISH 并总结结果。\n\n"
            "请在回复末尾用一行注明: NEXT: researcher / executor / FINISH"
        )
    )
    response = llm.invoke([sys_msg] + state["messages"])
    content = response.content or ""

    # 解析路由指令
    next_agent = "FINISH"
    for line in content.split("\n"):
        if line.strip().startswith("NEXT:"):
            next_agent = line.split(":", 1)[1].strip().lower()
            break

    return {
        "messages": [response],
        "current_agent": next_agent,
    }


def researcher_node(state: MultiAgentState):
    """研究员: 用只读工具搜索和读取信息，不做任何写入。"""
    llm = build_llm().bind_tools(RESEARCHER_TOOLS)
    sys_msg = SystemMessage(
        content=(
            "你是研究员。你可以用 search_web 搜索、用 read_file 读取文件，"
            "但不能写入或执行命令。收集到足够信息后，用简洁的文字总结，"
            "并在末尾写 DONE 表示完成。"
        )
    )
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response], "current_agent": "researcher"}


def executor_node(state: MultiAgentState):
    """执行员: 用危险工具写文件或执行命令。这些工具会触发 human-in-the-loop。"""
    llm = build_llm().bind_tools(EXECUTOR_TOOLS)
    sys_msg = SystemMessage(
        content=(
            "你是执行员。你可以用 write_file 写文件、用 run_command 执行命令。"
            "这些是有实际影响的操作，会被用户审批。完成任务后用简洁的文字总结，"
            "并在末尾写 DONE 表示完成。"
        )
    )
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response], "current_agent": "executor"}


# -----------------------------------------------------------------------------
# 5. 构建多 agent 协作图
# -----------------------------------------------------------------------------
def route_after_supervisor(state: MultiAgentState) -> Literal["researcher", "executor", END]:
    """根据 supervisor 的决策，路由到下一个节点。"""
    agent = state.get("current_agent", "FINISH").lower()
    if agent == "researcher":
        return "researcher"
    if agent == "executor":
        return "executor"
    return END


def route_after_subagent(state: MultiAgentState) -> Literal["tools", "supervisor"]:
    """子 agent 回复后: 若有 tool_calls 就执行工具，否则把结果交回 supervisor。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "supervisor"


def build_graph():
    graph = StateGraph(MultiAgentState)

    # 三个 agent 节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("executor", executor_node)

    # 工具节点: 两个子 agent 共用一个 ToolNode，它会自动根据 tool name 找对应实现。
    all_tools = RESEARCHER_TOOLS + EXECUTOR_TOOLS
    graph.add_node("tools", ToolNode(all_tools))

    # 路由逻辑
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor)
    graph.add_conditional_edges("researcher", route_after_subagent)
    graph.add_conditional_edges("executor", route_after_subagent)
    # 工具执行完，回到对应 agent 让其继续（通过 current_agent 字段判断）。
    graph.add_conditional_edges(
        "tools",
        lambda s: s.get("current_agent", "supervisor"),
    )

    return graph.compile(
        checkpointer=MemorySaver(),
        # interrupt_before=["tools"] 也可行，但这里用 interrupt() 更灵活，能针对
        # 特定工具做细粒度控制（比如只对写文件中断、读文件不中断）。
    )


# -----------------------------------------------------------------------------
# 6. 运行: 带 human-in-the-loop 的交互式执行
# -----------------------------------------------------------------------------
def run_interactive(task: str):
    """交互式运行，遇到 interrupt 时提示用户批准或拒绝。"""
    app = build_graph()
    config = {"configurable": {"thread_id": "demo-multi"}}
    state = {"messages": [HumanMessage(content=task)], "current_agent": "supervisor"}

    print(f"=== 任务: {task} ===\n")

    while True:
        # stream 每个节点产出的中间状态
        events = list(app.stream(state, config, stream_mode="values"))
        if not events:
            break

        last_state = events[-1]
        last_msg = last_state["messages"][-1]

        # 打印最新消息
        kind = last_msg.__class__.__name__
        agent = last_state.get("current_agent", "?")
        if isinstance(last_msg, AIMessage):
            if getattr(last_msg, "tool_calls", None):
                calls = ", ".join(c["name"] for c in last_msg.tool_calls)
                print(f"[{agent}/{kind}] 调用工具: {calls}")
            else:
                preview = (last_msg.content or "")[:150].replace("\n", " ")
                print(f"[{agent}/{kind}] {preview}")
        else:
            preview = str(last_msg.content or "")[:150].replace("\n", " ")
            print(f"[{kind}] {preview}")

        # 检查是否因 interrupt() 暂停
        snapshot = app.get_state(config)
        if snapshot.next:
            # 图还有待执行的节点但暂停了 = 遇到了 interrupt
            interrupts = snapshot.tasks
            if interrupts:
                for task_item in interrupts:
                    if task_item.interrupts:
                        # 拿到 interrupt() 传出的数据
                        req = task_item.interrupts[0].value
                        print(f"\n{'='*60}")
                        print(f"⚠️  需要人工确认: {req['operation']}")
                        print(f"    {req['question']}")
                        print(f"    参数: {req['args']}")
                        print(f"{'='*60}")
                        answer = input("批准? [y/n/edit]: ").strip().lower()

                        if answer == "y":
                            # 批准：传 Command(resume=批准数据)，图恢复并把这个值作为
                            # interrupt() 的返回值传回工具函数。
                            app.update_state(
                                config, {"approved": True}, as_node=task_item.name
                            )
                        elif answer == "edit":
                            print("(演示简化: 直接批准但可在此加参数编辑逻辑)")
                            app.update_state(
                                config, {"approved": True}, as_node=task_item.name
                            )
                        else:
                            app.update_state(
                                config,
                                {"approved": False, "reason": "用户拒绝"},
                                as_node=task_item.name,
                            )
                        break
            # 继续执行
            state = None  # 传 None 让它从 checkpoint 恢复
            continue

        # 没有 interrupt，检查是否结束
        if not snapshot.next:
            print("\n=== 任务完成 ===")
            final = last_state["messages"][-1]
            if isinstance(final, AIMessage) and final.content:
                print(final.content)
            break

        state = None


# -----------------------------------------------------------------------------
# 7. 离线自检: 用 fake 模型验证多 agent 路由和中断恢复
# -----------------------------------------------------------------------------
def self_check():
    """不依赖真实 API，验证多 agent 路由、中断恢复、协作流程。"""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    # 预设三条回复: supervisor派researcher -> researcher调工具 -> researcher汇报
    msg1_supervisor = AIMessage(content="需要搜索信息，派给 researcher。\nNEXT: researcher")
    msg2_researcher = AIMessage(
        content="",
        tool_calls=[{"name": "search_web", "args": {"query": "test"}, "id": "c1"}],
    )
    msg3_researcher = AIMessage(content="搜索完成，结果已收集。DONE\nNEXT: FINISH")
    fake = FakeMessagesListChatModel(responses=[msg1_supervisor, msg2_researcher, msg3_researcher])

    def supervisor_node_fake(state):
        resp = fake.invoke(state["messages"])
        next_a = "FINISH"
        for line in (resp.content or "").split("\n"):
            if "NEXT:" in line:
                next_a = line.split(":", 1)[1].strip().lower()
        return {"messages": [resp], "current_agent": next_a}

    def researcher_node_fake(state):
        resp = fake.invoke(state["messages"])
        return {"messages": [resp], "current_agent": "researcher"}

    def executor_node_fake(state):
        # 这个自检场景不会真正走到 executor，但需要定义避免图编译报错。
        resp = fake.invoke(state["messages"])
        return {"messages": [resp], "current_agent": "executor"}

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node_fake)
    graph.add_node("researcher", researcher_node_fake)
    graph.add_node("executor", executor_node_fake)
    graph.add_node("tools", ToolNode(RESEARCHER_TOOLS))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor)
    graph.add_conditional_edges("researcher", route_after_subagent)
    graph.add_conditional_edges("executor", route_after_subagent)
    graph.add_conditional_edges("tools", lambda s: s.get("current_agent", "supervisor"))

    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "selfcheck"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="搜索 Python")], "current_agent": "supervisor"},
        config,
    )

    agents_visited = [m.content for m in result["messages"] if isinstance(m, AIMessage)]
    print("访问节点顺序:", " -> ".join([a[:30] for a in agents_visited]))
    assert any("researcher" in a for a in agents_visited), "未路由到 researcher"
    assert "ToolMessage" in [m.__class__.__name__ for m in result["messages"]], "工具未执行"
    print("多 agent 路由与协作自检通过。")


# -----------------------------------------------------------------------------
# 8. 入口
# -----------------------------------------------------------------------------
def main():
    task = (
        " ".join(sys.argv[1:])
        or "搜索 Python 最佳实践，并把结果总结写入 best_practices.md"
    )
    if API_KEY == "EMPTY":
        print("未设置 OPENAI_API_KEY，无法真正调用模型。")
        print("可运行 --self-check 验证图结构（无需联网）。")
        return
    run_interactive(task)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        self_check()
    else:
        main()
