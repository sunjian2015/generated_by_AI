"""只依赖标准库 + openai SDK 的最小「代码 Agent」示例。

运行:
    conda activate torch          # 或任何装了 openai 库的环境
    pip install openai            # 唯一的第三方依赖
    set OPENAI_API_KEY=sk-xxx     # Windows PowerShell: $env:OPENAI_API_KEY="sk-xxx"
    python code_agent_minimal.py "把 README 里的运行步骤翻译成英文写入 README_en.md"

这份代码关注 Agent 的骨架而不是花哨的功能：一个 LLM 作为「大脑」，
通过 tool use（function calling）读文件、写文件、列目录、执行 shell 命令，
在一个循环里反复「思考 -> 调用工具 -> 观察结果」，直到任务完成。

后端使用 OpenAI 兼容协议，通过 OPENAI_BASE_URL 可指向 DeepSeek、Moonshot、
本地 vLLM / Ollama(/v1) 等任意兼容端点，无需改代码。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI


# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
# 所有可调项都从环境变量读取，方便切换不同的兼容后端而不改代码。
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None 时用 openai 官方地址
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")  # 本地端点常用占位符

# Agent 主循环最多迭代多少轮。每一轮 = 一次模型调用 + （可能的）工具执行。
# 设上限是最重要的安全阀，避免模型陷入死循环无限烧钱。
MAX_STEPS = 25

# 工作根目录。所有文件读写都被限制在这个目录内，防止 Agent 意外碰到
# 系统其它位置的文件。默认就是当前目录。
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", ".")).resolve()

# 单条 shell 命令的超时（秒），防止一条命令卡死整个 Agent。
SHELL_TIMEOUT = 60

# 是否在执行 shell 命令前要求人工确认。执行命令是不可逆、高风险操作，
# 默认开启确认。设 AGENT_AUTO_APPROVE=1 可全自动（仅在信任环境使用）。
REQUIRE_APPROVAL = os.environ.get("AGENT_AUTO_APPROVE", "0") != "1"


# 系统提示定义 Agent 的角色、工作方式和约束。这是决定 Agent 行为质量的
# 关键之一：工具再多，模型不知道何时/如何用也没意义。
SYSTEM_PROMPT = f"""你是一个运行在命令行里的代码 Agent。你的工作目录是: {WORKDIR}

你可以调用工具来观察和修改这个目录里的代码，一步步完成用户交给你的任务。

工作方式:
- 先用 list_dir / read_file 了解现状，再动手修改，不要凭空猜测文件内容。
- 修改文件用 write_file；它会覆盖整个文件，所以先读出原内容再改。
- 需要运行测试、构建、git 等命令时用 run_shell。
- 每次只做一小步，根据工具返回的结果决定下一步。
- 任务完成后，直接用自然语言总结你做了什么，不要再调用工具。

约束:
- 所有路径都相对于工作目录，禁止访问工作目录以外的文件。
- 不要执行有破坏性或不可逆的危险命令（如递归删除、格式化磁盘）。
- 保持改动最小化，只做任务要求的事。
"""


# -----------------------------------------------------------------------------
# 2. 工具定义（给模型看的 JSON Schema）
# -----------------------------------------------------------------------------
# 这份 schema 是模型「知道自己有哪些能力」的唯一依据。description 写得越清楚，
# 模型选对工具、填对参数的概率越高。参数结构遵循 JSON Schema。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出某个目录下的文件和子目录。用于了解项目结构。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于工作目录的路径，'.' 表示工作目录本身。",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文本文件的完整内容。修改文件前应先读。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于工作目录的文件路径。",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把内容写入文件（覆盖原文件，不存在则创建）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于工作目录的文件路径。",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文本内容。",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "在工作目录下执行一条 shell 命令并返回其输出。用于测试、构建、git 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的完整命令行。",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


# -----------------------------------------------------------------------------
# 3. 工具执行与安全防护
# -----------------------------------------------------------------------------
def resolve_in_workdir(path):
    """把相对路径解析为绝对路径，并确保它仍落在 WORKDIR 内。

    这是路径穿越（path traversal）防护：模型可能生成 '../../etc/passwd'
    之类的路径，我们必须拒绝任何逃出工作目录的访问。
    """
    target = (WORKDIR / path).resolve()
    if target != WORKDIR and WORKDIR not in target.parents:
        raise ValueError(f"路径 {path!r} 超出了工作目录，已拒绝。")
    return target


def tool_list_dir(path="."):
    target = resolve_in_workdir(path)
    if not target.is_dir():
        return f"错误: {path} 不是一个目录。"
    entries = []
    for child in sorted(target.iterdir()):
        entries.append(child.name + ("/" if child.is_dir() else ""))
    return "\n".join(entries) if entries else "(空目录)"


def tool_read_file(path):
    target = resolve_in_workdir(path)
    if not target.is_file():
        return f"错误: {path} 不是一个文件。"
    text = target.read_text(encoding="utf-8", errors="replace")
    # 截断超长文件，避免单次工具结果撑爆上下文窗口。
    max_chars = 20000
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (已截断，共 {len(text)} 字符)"
    return text


def tool_write_file(path, content):
    target = resolve_in_workdir(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字符）。"


def tool_run_shell(command):
    """执行 shell 命令。默认在执行前请求人工确认，这是最重要的安全闸门。"""
    if REQUIRE_APPROVAL:
        print(f"\n[需要确认] Agent 想执行命令:\n    {command}")
        answer = input("允许执行? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            return "用户拒绝执行该命令。"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"错误: 命令超过 {SHELL_TIMEOUT}s 超时。"

    # 把 stdout / stderr / 退出码都返回给模型，让它自己判断成功与否。
    parts = [f"退出码: {completed.returncode}"]
    if completed.stdout:
        parts.append("stdout:\n" + completed.stdout)
    if completed.stderr:
        parts.append("stderr:\n" + completed.stderr)
    return "\n".join(parts)


# 把工具名映射到实现函数，供主循环按名分发。
TOOL_IMPLEMENTATIONS = {
    "list_dir": tool_list_dir,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_shell": tool_run_shell,
}


def execute_tool(name, arguments):
    """按名分发并执行一个工具，把任何异常转成给模型看的错误字符串。

    关键点: 工具执行失败不应让整个 Agent 崩溃。把错误作为普通工具结果
    喂回模型，模型往往能读懂错误并自行纠正（比如换个路径重试）。
    """
    func = TOOL_IMPLEMENTATIONS.get(name)
    if func is None:
        return f"错误: 未知工具 {name!r}。"
    try:
        return func(**arguments)
    except Exception as exc:  # noqa: BLE001  教学示例，统一兜底
        return f"工具 {name} 执行出错: {exc}"


# -----------------------------------------------------------------------------
# 4. LLM 调用封装
# -----------------------------------------------------------------------------
def build_client():
    """构造 OpenAI 兼容客户端。base_url 为空时走官方地址。"""
    kwargs = {"api_key": API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def call_model(client, messages):
    """调用一次 chat completions，带上工具 schema，返回 assistant message。

    tool_choice='auto' 让模型自己决定是回复文本还是调用工具，这正是
    Agent「自主性」的来源。
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0.0,  # 代码任务追求确定性，降低随机
    )
    return response.choices[0].message


# -----------------------------------------------------------------------------
# 5. Agent 主循环（ReAct 风格）
# -----------------------------------------------------------------------------
def run_agent(task):
    """Agent 的核心：观察 -> 思考 -> 行动 的循环。

    每一轮:
        1) 把完整对话历史发给模型；
        2) 模型要么返回工具调用，要么返回最终文本回复；
        3) 若是工具调用，逐个执行并把结果作为 role='tool' 消息追加回历史，
           然后进入下一轮，让模型基于新观察继续；
        4) 若是纯文本回复，说明模型认为任务完成，退出循环。

    对话历史（messages）就是 Agent 的「短期记忆」：它把每一步的思考、
    工具调用和工具结果都串起来，模型据此保持连贯。
    """
    client = build_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        message = call_model(client, messages)

        # 把 assistant 这一轮的输出原样追加进历史（含 tool_calls），
        # 否则后续的 tool 结果消息会与它对不上号。
        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls
        if not tool_calls:
            # 没有工具调用 = 模型给出了最终答复，任务结束。
            print(f"\n=== Agent 完成（共 {step} 步）===")
            print(message.content or "(无文本输出)")
            return

        # 依次执行本轮模型请求的每个工具调用。
        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            print(f"\n[step {step}] 调用工具 {name} 参数={arguments}")
            result = execute_tool(name, arguments)
            preview = result if len(result) <= 500 else result[:500] + " ...(截断)"
            print(f"[step {step}] 结果:\n{preview}")

            # 工具结果必须通过 tool_call_id 关联回对应的调用。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    print(f"\n=== 达到最大步数 {MAX_STEPS}，强制停止 ===")


# -----------------------------------------------------------------------------
# 6. CLI 入口
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="最小代码 Agent 示例")
    parser.add_argument(
        "task",
        nargs="?",
        help="要交给 Agent 的任务描述；省略时进入交互输入。",
    )
    args = parser.parse_args()

    task = args.task or input("请输入任务: ").strip()
    if not task:
        print("未提供任务，退出。")
        sys.exit(1)

    print(f"模型: {MODEL}  后端: {BASE_URL or 'OpenAI 官方'}")
    print(f"工作目录: {WORKDIR}")
    print(f"命令确认: {'开启' if REQUIRE_APPROVAL else '关闭(全自动)'}")
    run_agent(task)


if __name__ == "__main__":
    main()
