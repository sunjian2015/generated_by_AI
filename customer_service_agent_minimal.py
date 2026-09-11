"""只依赖标准库 + openai SDK 的最小「客服 Agent」示例。

运行:
    conda activate torch
    pip install openai
    set OPENAI_API_KEY=sk-xxx        # PowerShell: $env:OPENAI_API_KEY="sk-xxx"
    python customer_service_agent_minimal.py

和 code_agent_minimal.py 的区别在于「形态」:
    - 代码 Agent 是「给一个任务，自主跑到完成」的单轮任务型;
    - 客服 Agent 是「多轮对话」型: 用户说一句 -> Agent 内部可能查若干次业务
      工具 -> 回一句 -> 等用户下一句，循环往复。

因此本文件有两层循环:
    外层 = 多轮对话 REPL（等用户输入）;
    内层 = 单个用户回合内的 tool-use 循环（agent 自主查数据直到能回答）。

为不触碰真实用户数据，FAQ 和订单都是内置的假数据。接真实系统时，只需把
第 3 节里几个工具函数替换成真实 API/数据库调用即可，其余骨架不用动。
"""

import json
import os
import sys

from openai import OpenAI


# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None 时用 openai 官方地址
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

# 单个用户回合内，agent 最多连续调用多少次工具后必须给出回复。
# 防止模型在查数据时陷入死循环。
MAX_TOOL_STEPS = 8

# 公司/产品名，注入系统提示，方便一处修改。
COMPANY = os.environ.get("AGENT_COMPANY", "云上优选商城")


# 系统提示定义客服的人设、能力边界和行为准则。客服场景里「边界」尤其重要:
# 不能乱承诺、不能泄露他人信息、拿不准就转人工。
SYSTEM_PROMPT = f"""你是「{COMPANY}」的在线客服助手。你的目标是高效、友好地解决用户的问题。

你可以调用工具来查询真实业务数据，不要凭空编造订单、物流或政策信息:
- search_faq: 查询常见问题库（退换货政策、发票、运费等通用问题优先查这里）。
- get_order: 用订单号查询订单详情。
- get_logistics: 用订单号查询物流状态。
- create_refund: 为订单发起退款申请。
- escalate_to_human: 转接人工客服。

行为准则:
- 回答政策类问题前先查 FAQ，不要靠记忆编造。
- 涉及具体订单时，务必先用 get_order 核实，再基于真实数据回答。
- 发起退款（create_refund）属于会产生实际影响的操作: 执行前一定要向用户
  确认订单号和退款原因，用户明确同意后才调用。
- 遇到以下情况直接转人工（escalate_to_human）: 投诉升级、要求赔偿、你无法
  用工具解决的问题、或用户明确要求人工。
- 不要索取或复述银行卡号、密码、验证码等敏感信息。
- 保持简洁礼貌，一次聚焦解决一个问题。
"""


# -----------------------------------------------------------------------------
# 2. 模拟业务数据（真实系统里替换为数据库 / 微服务）
# -----------------------------------------------------------------------------
# FAQ 知识库: 关键词 -> 答案。真实场景通常用向量检索，这里用关键词命中演示。
FAQ_DB = {
    "退货": "支持 7 天无理由退货。商品需保持完好、不影响二次销售。生鲜、定制类商品除外。",
    "换货": "签收后 15 天内可申请换货，非质量问题的换货运费由买家承担。",
    "运费": "单笔订单满 99 元包邮，未满收取 10 元运费；港澳台及偏远地区另计。",
    "发票": "支持开具电子普通发票和增值税专用发票，可在订单详情页「申请开票」中操作。",
    "配送时间": "现货商品 48 小时内发货，一般 2-4 天送达；预售商品以商品页标注为准。",
}

# 订单库: 订单号 -> 订单信息。数据均为虚构，不含真实个人信息。
ORDER_DB = {
    "A1001": {
        "status": "已发货",
        "item": "无线蓝牙耳机",
        "amount": 199.0,
        "paid": True,
        "buyer": "张*",
    },
    "A1002": {
        "status": "待发货",
        "item": "机械键盘",
        "amount": 359.0,
        "paid": True,
        "buyer": "李*",
    },
    "A1003": {
        "status": "已签收",
        "item": "保温杯",
        "amount": 89.0,
        "paid": True,
        "buyer": "王*",
    },
}

# 物流库: 订单号 -> 物流轨迹。
LOGISTICS_DB = {
    "A1001": ["商品已出库(北京仓)", "运输中: 已到达济南转运中心", "派送中: 聊城市"],
    "A1003": ["商品已出库(上海仓)", "运输中", "已签收: 本人签收"],
}

# 用一个内存字典记录本次会话已发起的退款，避免重复提交（模拟幂等）。
REFUNDS = {}


# -----------------------------------------------------------------------------
# 3. 工具定义与实现
# -----------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "在常见问题库中检索政策类问题的答案（退货、换货、运费、发票、配送等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户问题或关键词。"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "用订单号查询订单详情（状态、商品、金额等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，如 A1001。"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logistics",
            "description": "用订单号查询物流轨迹。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，如 A1001。"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": "为指定订单发起退款申请。这是会产生实际影响的操作，调用前必须已获得用户明确同意。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "要退款的订单号。"},
                    "reason": {"type": "string", "description": "退款原因。"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "将当前会话转接给人工客服。用于投诉升级、赔偿诉求或工具无法解决的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "给人工客服的问题摘要，方便快速接手。",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


def tool_search_faq(query):
    # 关键词命中演示。真实场景应换成向量检索 / 全文检索。
    hits = [f"【{kw}】{ans}" for kw, ans in FAQ_DB.items() if kw in query]
    if hits:
        return "\n".join(hits)
    # 没命中时返回全部条目，让模型自行判断相关性，也可改为返回「未找到」。
    catalog = "、".join(FAQ_DB.keys())
    return f"未直接命中。FAQ 可用主题: {catalog}。"


def tool_get_order(order_id):
    order_id = order_id.strip().upper()
    order = ORDER_DB.get(order_id)
    if not order:
        return f"未找到订单 {order_id}，请用户确认订单号是否正确。"
    return json.dumps({"order_id": order_id, **order}, ensure_ascii=False)


def tool_get_logistics(order_id):
    order_id = order_id.strip().upper()
    if order_id not in ORDER_DB:
        return f"未找到订单 {order_id}。"
    track = LOGISTICS_DB.get(order_id)
    if not track:
        return f"订单 {order_id} 暂无物流信息（可能尚未发货）。"
    return " -> ".join(track)


def tool_create_refund(order_id, reason):
    order_id = order_id.strip().upper()
    order = ORDER_DB.get(order_id)
    if not order:
        return f"退款失败: 未找到订单 {order_id}。"
    if order_id in REFUNDS:
        return f"订单 {order_id} 已有退款申请（{REFUNDS[order_id]}），无需重复提交。"
    # 简单业务规则示例: 已签收超期、未支付等可在此拦截。这里只做已支付校验。
    if not order.get("paid"):
        return f"退款失败: 订单 {order_id} 未完成支付，无需退款。"
    REFUNDS[order_id] = reason
    return (
        f"已为订单 {order_id}（{order['item']}，¥{order['amount']}）发起退款申请，"
        f"原因: {reason}。预计 1-3 个工作日原路退回。"
    )


def tool_escalate_to_human(summary):
    # 真实系统里这里会创建工单 / 转接会话。演示中仅返回确认。
    return f"已转接人工客服，工单摘要: {summary}。人工客服将尽快跟进。"


TOOL_IMPLEMENTATIONS = {
    "search_faq": tool_search_faq,
    "get_order": tool_get_order,
    "get_logistics": tool_get_logistics,
    "create_refund": tool_create_refund,
    "escalate_to_human": tool_escalate_to_human,
}


def execute_tool(name, arguments):
    """按名分发执行工具，异常统一兜底为给模型看的错误串。"""
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
    kwargs = {"api_key": API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def call_model(client, messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0.3,  # 客服回复要稳定，但比纯代码任务略保留一点自然度
    )
    return response.choices[0].message


# -----------------------------------------------------------------------------
# 5. 单回合处理: 内层 tool-use 循环
# -----------------------------------------------------------------------------
def handle_turn(client, messages):
    """处理用户的一句话: 反复调用工具直到模型给出面向用户的自然语言回复。

    messages 是整个会话历史（含 system、历次 user/assistant/tool 消息），
    即 Agent 的记忆。本函数会就地把新产生的消息追加进去，并返回最终回复文本。
    """
    for _ in range(MAX_TOOL_STEPS):
        message = call_model(client, messages)
        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls
        if not tool_calls:
            # 模型给出了面向用户的回复，本回合结束。
            return message.content or "(客服未给出内容)"

        # 执行本轮所有工具调用，结果回填历史后继续内层循环。
        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            print(f"  · [调用] {name}({arguments})")
            result = execute_tool(name, arguments)
            print(f"  · [结果] {result}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    # 工具步数用尽仍未收敛，兜底转人工，避免卡死。
    return "这个问题稍复杂，我先为您转接人工客服跟进。"


# -----------------------------------------------------------------------------
# 6. 多轮对话 REPL 入口
# -----------------------------------------------------------------------------
def main():
    client = build_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"模型: {MODEL}  后端: {BASE_URL or 'OpenAI 官方'}")
    print(f"欢迎来到「{COMPANY}」在线客服（输入 exit / quit 结束）")
    print("提示: 可试试订单号 A1001 / A1002 / A1003，或问退货、运费等政策。\n")
    print("客服: 您好，很高兴为您服务，请问有什么可以帮您？")

    while True:
        try:
            user_input = input("\n您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话结束，感谢咨询。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("客服: 感谢您的咨询，再见！")
            break

        messages.append({"role": "user", "content": user_input})
        reply = handle_turn(client, messages)
        print(f"客服: {reply}")


if __name__ == "__main__":
    main()
