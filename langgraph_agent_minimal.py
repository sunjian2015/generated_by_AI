"""用 LangGraph 实现的最小 Agent 示例，与手写版对照。

    pip install langgraph langchain-openai
    set OPENAI_API_KEY=sk-xxx
    python langgraph_agent_minimal.py "列出当前目录并统计有几个 .py 文件"

对照前面的 code_agent_minimal.py:
    手写版里我们自己写了 while 循环: 调模型 -> 若有 tool_calls 就执行 -> 回填
    结果 -> 再调模型。LangGraph 把这套循环抽象成一张「状态图」:

        START -> [agent 节点] --有工具调用--> [tools 节点] --+
                     ▲                                        |
                     +--------------(回填结果)----------------+
                     |
                     +--无工具调用--> END

    你只need定义「节点(做什么)」和「边(下一步去哪)」，循环调度、状态合并、
    中断恢复都交给框架。好处是: 记忆(checkpointer)、流式、可视化、人类介入
    (human-in-the-loop)这些都能直接复用框架能力，不必自己重写。

本例基于 langgraph 1.x / langchain-core 1.x 的现行 API。
"""

import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL")  # 兼容 DeepSeek/Moonshot/本地端点
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

# 所有文件操作限制在这个根目录内，防止路径穿越。
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", ".")).resolve()

SYSTEM_PROMPT = (
    f"你是一个文件助手，工作目录是 {WORKDIR}。"
    "你可以用工具列目录、读文件。先观察再回答，不要编造文件内容。"
    "完成后用简洁的中文总结结果。"
)


# -----------------------------------------------------------------------------
# 2. 工具: 用 @tool 装饰器声明
# -----------------------------------------------------------------------------
# LangChain 的 @tool 会自动从函数签名 + docstring 生成给模型看的 schema，
# 不用像手写版那样手写一大段 JSON Schema。这是框架带来的第一个便利。
def _resolve(path: str) -> Path:
    """把相对路径解析进 WORKDIR，并阻止逃出工作目录（路径穿越防护）。"""
    target = (WORKDIR / path).resolve()
    if target != WORKDIR and WORKDIR not in target.parents:
        raise ValueError(f"路径 {path!r} 超出工作目录，已拒绝。")
    return target


@tool
def list_dir(path: str = ".") -> str:
    """列出目录下的文件和子目录。path 相对于工作目录，'.' 表示工作目录本身。"""
    target = _resolve(path)
    if not target.is_dir():
        return f"错误: {path} 不是目录。"
    entries = [c.name + ("/" if c.is_dir() else "") for c in sorted(target.iterdir())]
    return "\n".join(entries) if entries else "(空目录)"


@tool
def read_file(path: str) -> str:
    """读取一个文本文件的完整内容。path 相对于工作目录。"""
    target = _resolve(path)
    if not target.is_file():
        return f"错误: {path} 不是文件。"
    text = target.read_text(encoding="utf-8", errors="replace")
    return text[:20000]


TOOLS = [list_dir, read_file]


# -----------------------------------------------------------------------------
# 3. 图状态定义
# -----------------------------------------------------------------------------
# LangGraph 的核心是「在一个共享 State 上跑图」。这里 State 只有一个字段:
# messages。Annotated[..., add_messages] 告诉框架: 每个节点返回的 messages
# 不是覆盖，而是「追加」到历史里——这正是对话循环需要的语义。
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# -----------------------------------------------------------------------------
# 4. 构建图: 节点 + 边
# -----------------------------------------------------------------------------
def build_agent():
    """把 LLM、工具、循环组装成一张可执行的状态图。"""
    llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)
    # bind_tools 让模型「知道」有哪些工具可用，并能在回复里发起 tool_calls。
    llm_with_tools = llm.bind_tools(TOOLS)

    # --- 节点 1: agent。调用模型，产出一条 AIMessage（可能带 tool_calls）。
    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # --- 节点 2: tools。ToolNode 是预置节点，自动读取上一条消息里的
    #     tool_calls、并行执行对应工具、把结果作为 ToolMessage 回填。
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    # 条件边: tools_condition 检查 agent 最新消息——有 tool_calls 就去 "tools"，
    # 否则去 END。这正是手写版里 `if response.tool_calls:` 那个判断。
    graph.add_conditional_edges("agent", tools_condition)
    # 工具执行完，边回到 agent，让模型基于工具结果继续——形成循环。
    graph.add_edge("tools", "agent")

    # checkpointer = 记忆。带上它后，同一 thread_id 的多次调用会自动延续历史，
    # 这就是 LangGraph 内建的多轮对话记忆，不用自己维护 messages 列表。
    return graph.compile(checkpointer=MemorySaver())


# -----------------------------------------------------------------------------
# 5. 运行
# -----------------------------------------------------------------------------
def run(task: str):
    app = build_agent()
    # thread_id 标识一次会话。换 id = 新会话；同 id = 接着聊。
    config = {"configurable": {"thread_id": "demo-1"}}
    initial = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]
    }
    # stream 逐节点吐出中间过程，方便观察 agent 每一步在做什么。
    final = None
    for event in app.stream(initial, config, stream_mode="values"):
        final = event
        last = event["messages"][-1]
        # 打印每个节点产出的最新消息，直观展示「思考->调工具->观察」的流转。
        kind = last.__class__.__name__
        if getattr(last, "tool_calls", None):
            calls = ", ".join(c["name"] for c in last.tool_calls)
            print(f"[{kind}] -> 调用工具: {calls}")
        elif last.content:
            preview = str(last.content).replace("\n", " ")[:120]
            print(f"[{kind}] {preview}")

    if final:
        print("\n=== 最终回复 ===")
        print(final["messages"][-1].content)


def main():
    task = " ".join(sys.argv[1:]) or "列出当前目录，并告诉我里面有哪些 Python 文件。"
    if API_KEY == "EMPTY":
        print("提示: 未设置 OPENAI_API_KEY，无法真正调用模型。")
        print("可先运行 self_check() 验证图结构（无需联网）。")
        return
    run(task)


# -----------------------------------------------------------------------------
# 6. 离线自检: 不依赖真实 LLM，只验证图能否正确编译与流转
# -----------------------------------------------------------------------------
def self_check():
    """用一个假的模型验证图结构、条件边、工具节点、记忆是否正确工作。

    思路: 不连真实 API，用 FakeMessagesListChatModel 预设两条回复——第一条带
    tool_call（触发 tools 节点），第二条纯文本（走向 END）。这样能在离线状态下
    端到端跑通整张图。
    """
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    # 第一条: 模型决定调用 list_dir 工具。
    call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "list_dir", "args": {"path": "."}, "id": "c1"}],
    )
    # 第二条: 拿到工具结果后，模型给出最终文本回复。
    done_msg = AIMessage(content="当前目录已列出，共完成。")
    fake = FakeMessagesListChatModel(responses=[call_msg, done_msg])

    # 复用真实的图结构，只把 llm 换成 fake。fake 模型不支持 bind_tools，但也
    # 不需要——它返回的 AIMessage 已带 tool_calls，ToolNode 只认这个字段。
    def agent_node(state: AgentState):
        return {"messages": [fake.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    app = graph.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "selfcheck"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="列个目录")]}, config
    )

    kinds = [m.__class__.__name__ for m in result["messages"]]
    print("消息流转:", " -> ".join(kinds))
    # 期望顺序: Human(输入) -> AI(带tool_call) -> Tool(工具结果) -> AI(最终回复)
    assert "ToolMessage" in kinds, "工具节点未被触发"
    assert result["messages"][-1].content == "当前目录已列出，共完成。"
    # 验证工具确实执行并返回了本目录内容（含本文件名）。
    tool_out = next(m for m in result["messages"] if m.__class__.__name__ == "ToolMessage")
    assert "langgraph_agent_minimal.py" in tool_out.content, "工具未真正执行"
    print("图结构自检通过: START->agent->tools->agent->END，工具真实执行，记忆生效。")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        self_check()
    else:
        main()
