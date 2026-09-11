"""微信公众号（Official Account）自动回复 Agent —— 最小可运行服务端。

只依赖: 标准库(http.server) + cryptography(AES) + openai。无需 Flask。

    pip install openai cryptography
    set OPENAI_API_KEY=sk-xxx
    set WECHAT_TOKEN=your_token                # 与公众号后台「服务器配置」一致
    set WECHAT_AES_KEY=43位EncodingAESKey       # 安全模式必填，明文模式可省
    set WECHAT_APPID=wxxxxxxxxxxxx              # 安全模式必填
    python wechat_mp_agent_minimal.py           # 默认监听 0.0.0.0:8080/wechat

架构要点: 微信这层只是「消息进出适配器」，真正的对话大脑复用前面 agent 的
LLM 调用。难点全在公众号协议本身:

    1) 服务器验证(GET): 校验 signature=sha1(sort(token,timestamp,nonce))。
    2) 消息加解密(POST 安全模式): 微信把消息 AES-256-CBC 加密后推来，回复也要
       加密。加解密格式是微信特有的 WXBizMsgCrypt。
    3) 5 秒超时: 微信要求 5s 内响应，否则重试 3 次、最终报错。LLM 通常慢于 5s，
       所以这里用「后台线程算 + MsgId 去重缓存 + 短等待」的经典模式来兜。

生产级更稳的做法是改用「客服消息」异步接口(见文末 send_customer_message 说明)，
但那需要已认证服务号 + access_token，这里给出可离线自测的最小骨架。
"""

import base64
import hashlib
import os
import struct
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from openai import OpenAI


# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

WECHAT_TOKEN = os.environ.get("WECHAT_TOKEN", "test_token")
WECHAT_AES_KEY = os.environ.get("WECHAT_AES_KEY", "")  # 43 位 EncodingAESKey
WECHAT_APPID = os.environ.get("WECHAT_APPID", "")

HOST = os.environ.get("WECHAT_HOST", "0.0.0.0")
PORT = int(os.environ.get("WECHAT_PORT", "8080"))
PATH = os.environ.get("WECHAT_PATH", "/wechat")

# 每个用户保留多少轮对话作为上下文（短期记忆）。
MEMORY_TURNS = 6
# 单次请求最多等待 LLM 多少秒。必须 < 微信的 5s 硬超时，留出网络余量。
REPLY_WAIT_SECONDS = 4.5

SYSTEM_PROMPT = os.environ.get(
    "WECHAT_SYSTEM_PROMPT",
    "你是一个微信公众号的智能助手，用简洁、友好的中文回答用户问题。"
    "每条回复尽量控制在 200 字以内，适合手机阅读。",
)


# -----------------------------------------------------------------------------
# 2. WXBizMsgCrypt: 签名校验 + AES 消息加解密
# -----------------------------------------------------------------------------
def check_signature(token, signature, timestamp, nonce):
    """服务器验证 / 消息签名校验。

    微信规则: 把 token、timestamp、nonce（安全模式再加 encrypt）按字典序排序、
    拼接后取 sha1。相等即证明请求确实来自微信、且未被篡改。
    """
    items = sorted([token, timestamp, nonce])
    digest = hashlib.sha1("".join(items).encode()).hexdigest()
    return digest == signature


def check_msg_signature(token, msg_signature, timestamp, nonce, encrypt):
    items = sorted([token, timestamp, nonce, encrypt])
    digest = hashlib.sha1("".join(items).encode()).hexdigest()
    return digest == msg_signature


class MessageCrypto:
    """微信安全模式的 AES-256-CBC 加解密。

    EncodingAESKey 是 43 个字符，补一个 '=' 后 base64 解码得到 32 字节 AESKey；
    IV 取 AESKey 的前 16 字节。明文封装格式（微信规定）:

        random(16B) + msg_len(4B, 网络字节序) + msg + appid

    并用块大小 32 的 PKCS7 填充。
    """

    def __init__(self, aes_key_b64, appid):
        if not aes_key_b64:
            raise ValueError("安全模式需要 WECHAT_AES_KEY(EncodingAESKey)。")
        self.key = base64.b64decode(aes_key_b64 + "=")
        if len(self.key) != 32:
            raise ValueError("AESKey 解码后必须为 32 字节。")
        self.iv = self.key[:16]
        self.appid = appid

    def _cipher(self):
        return Cipher(algorithms.AES(self.key), modes.CBC(self.iv))

    @staticmethod
    def _pkcs7_pad(data, block=32):
        pad = block - (len(data) % block)
        return data + bytes([pad]) * pad

    @staticmethod
    def _pkcs7_unpad(data):
        return data[: -data[-1]]

    def encrypt(self, plaintext_xml):
        """把要回复的明文 XML 加密为 base64 密文。"""
        raw = plaintext_xml.encode("utf-8")
        content = (
            os.urandom(16)
            + struct.pack(">I", len(raw))
            + raw
            + self.appid.encode("utf-8")
        )
        padded = self._pkcs7_pad(content)
        enc = self._cipher().encryptor()
        ciphertext = enc.update(padded) + enc.finalize()
        return base64.b64encode(ciphertext).decode()

    def decrypt(self, encrypt_b64):
        """把微信推来的 base64 密文解密还原为明文 XML，并校验 appid。"""
        ciphertext = base64.b64decode(encrypt_b64)
        dec = self._cipher().decryptor()
        padded = dec.update(ciphertext) + dec.finalize()
        content = self._pkcs7_unpad(padded)
        xml_len = struct.unpack(">I", content[16:20])[0]
        xml = content[20 : 20 + xml_len].decode("utf-8")
        from_appid = content[20 + xml_len :].decode("utf-8")
        if self.appid and from_appid != self.appid:
            raise ValueError("appid 校验失败，消息可能被伪造。")
        return xml


# -----------------------------------------------------------------------------
# 3. 消息 XML 解析与被动回复构造
# -----------------------------------------------------------------------------
def parse_message(xml_text):
    """把微信消息 XML 解析成 dict。只取常用字段。"""
    root = ET.fromstring(xml_text)
    return {child.tag: (child.text or "") for child in root}


def build_text_reply(to_user, from_user, content):
    """构造被动回复的文本消息 XML（明文）。"""
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{int(time.time())}</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "</xml>"
    )


def build_encrypted_reply(crypto, plaintext_xml, timestamp, nonce):
    """安全模式: 把明文回复 XML 加密后再套一层带签名的信封。"""
    encrypt = crypto.encrypt(plaintext_xml)
    items = sorted([WECHAT_TOKEN, timestamp, nonce, encrypt])
    signature = hashlib.sha1("".join(items).encode()).hexdigest()
    return (
        "<xml>"
        f"<Encrypt><![CDATA[{encrypt}]]></Encrypt>"
        f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
        f"<TimeStamp>{timestamp}</TimeStamp>"
        f"<Nonce><![CDATA[{nonce}]]></Nonce>"
        "</xml>"
    )


# -----------------------------------------------------------------------------
# 4. Agent 核心: LLM 对话 + 每用户短期记忆
# -----------------------------------------------------------------------------
_client = None
_client_lock = threading.Lock()
# openid -> 该用户最近若干轮消息列表（不含 system）。
_user_memory = {}
_memory_lock = threading.Lock()


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            kwargs = {"api_key": API_KEY}
            if BASE_URL:
                kwargs["base_url"] = BASE_URL
            _client = OpenAI(**kwargs)
    return _client


def agent_reply(openid, user_text):
    """调用 LLM 生成回复，并维护该用户的短期对话记忆。"""
    with _memory_lock:
        history = _user_memory.get(openid, [])
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": user_text}]
        )

    response = get_client().chat.completions.create(
        model=MODEL, messages=messages, temperature=0.5
    )
    answer = response.choices[0].message.content or ""

    with _memory_lock:
        history = _user_memory.get(openid, [])
        history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
        # 只保留最近 MEMORY_TURNS 轮（每轮 2 条），控制上下文长度。
        _user_memory[openid] = history[-MEMORY_TURNS * 2 :]
    return answer


# -----------------------------------------------------------------------------
# 5. 5 秒超时处理: 后台计算 + MsgId 去重缓存
# -----------------------------------------------------------------------------
# 微信 5s 内收不到响应会重试（同一 MsgId，共 3 次）。我们首次收到消息即在后台
# 线程算 LLM 结果并按 MsgId 缓存；本次请求最多等 REPLY_WAIT_SECONDS。若算完就
# 立即回复，否则先回空串，等微信带同一 MsgId 重试时再命中缓存返回答案。
_answer_cache = {}  # msg_id -> {"done": bool, "answer": str}
_cache_lock = threading.Lock()


def _compute_async(msg_id, openid, user_text):
    try:
        answer = agent_reply(openid, user_text)
    except Exception as exc:  # noqa: BLE001  兜底，避免线程静默死掉
        answer = f"（服务暂时不可用: {exc}）"
    with _cache_lock:
        _answer_cache[msg_id] = {"done": True, "answer": answer}


def get_or_start_answer(msg_id, openid, user_text):
    """返回已缓存答案；若无则启动后台计算并等待至多 REPLY_WAIT_SECONDS。"""
    with _cache_lock:
        cached = _answer_cache.get(msg_id)
        if cached and cached["done"]:
            return cached["answer"]
        if cached is None:
            _answer_cache[msg_id] = {"done": False, "answer": ""}
            threading.Thread(
                target=_compute_async,
                args=(msg_id, openid, user_text),
                daemon=True,
            ).start()

    deadline = time.time() + REPLY_WAIT_SECONDS
    while time.time() < deadline:
        time.sleep(0.2)
        with _cache_lock:
            cached = _answer_cache.get(msg_id)
            if cached and cached["done"]:
                return cached["answer"]
    return None  # 本轮没算完，回空串等微信重试


# -----------------------------------------------------------------------------
# 6. HTTP 请求处理
# -----------------------------------------------------------------------------
def _use_safe_mode():
    """配了 AES key 和 appid 就走安全模式，否则明文模式。"""
    return bool(WECHAT_AES_KEY and WECHAT_APPID)


class WeChatHandler(BaseHTTPRequestHandler):
    def _send(self, body, status=200, content_type="application/xml"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        """服务器验证: 校验签名后原样返回 echostr。"""
        parsed = urlparse(self.path)
        if parsed.path != PATH:
            return self._send("not found", status=404, content_type="text/plain")
        q = parse_qs(parsed.query)
        signature = q.get("signature", [""])[0]
        timestamp = q.get("timestamp", [""])[0]
        nonce = q.get("nonce", [""])[0]
        echostr = q.get("echostr", [""])[0]
        if check_signature(WECHAT_TOKEN, signature, timestamp, nonce):
            self._send(echostr, content_type="text/plain")
        else:
            self._send("signature mismatch", status=403, content_type="text/plain")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != PATH:
            return self._send("not found", status=404, content_type="text/plain")
        q = parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")

        try:
            reply = self._handle_message(raw_body, q)
        except Exception as exc:  # noqa: BLE001  任何异常都回 success，避免微信报错
            print(f"[error] {exc}")
            reply = "success"
        self._send(reply)

    def _handle_message(self, raw_body, q):
        timestamp = q.get("timestamp", [str(int(time.time()))])[0]
        nonce = q.get("nonce", ["0"])[0]

        # 安全模式: 先验签再解密出真正的消息 XML。
        if _use_safe_mode():
            outer = parse_message(raw_body)
            encrypt = outer.get("Encrypt", "")
            msg_signature = q.get("msg_signature", [""])[0]
            if not check_msg_signature(
                WECHAT_TOKEN, msg_signature, timestamp, nonce, encrypt
            ):
                raise ValueError("msg_signature 校验失败。")
            crypto = MessageCrypto(WECHAT_AES_KEY, WECHAT_APPID)
            body_xml = crypto.decrypt(encrypt)
        else:
            crypto = None
            body_xml = raw_body

        msg = parse_message(body_xml)
        msg_type = msg.get("MsgType", "")
        from_user = msg.get("FromUserName", "")  # 用户 openid
        to_user = msg.get("ToUserName", "")  # 公众号原始 ID

        # 只处理文本消息；其它类型给个友好提示。事件(关注等)可在此扩展。
        if msg_type != "text":
            plain = build_text_reply(
                from_user, to_user, "目前我只能理解文字消息，请用文字描述您的问题～"
            )
            return self._wrap(crypto, plain, timestamp, nonce)

        user_text = msg.get("Content", "").strip()
        msg_id = msg.get("MsgId", "") or f"{from_user}-{msg.get('CreateTime','')}"

        answer = get_or_start_answer(msg_id, from_user, user_text)
        if answer is None:
            # 本轮没算完: 回空串（微信视为成功、不报错），等其重试命中缓存。
            return ""

        plain = build_text_reply(from_user, to_user, answer)
        return self._wrap(crypto, plain, timestamp, nonce)

    def _wrap(self, crypto, plaintext_xml, timestamp, nonce):
        if crypto is not None:
            return build_encrypted_reply(crypto, plaintext_xml, timestamp, nonce)
        return plaintext_xml

    def log_message(self, fmt, *args):
        # 精简默认访问日志。
        print("[http] " + (fmt % args))


# 关于「客服消息」异步方案（生产推荐，需已认证服务号）:
#   若 LLM 经常超过 5s，被动回复 + 重试仍可能来不及。更稳的做法是: 收到消息后
#   立刻回 "success"，随后用 access_token 调用客服消息接口主动推送答案:
#       POST https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token=XX
#       body = {"touser": openid, "msgtype":"text", "text":{"content": answer}}
#   access_token 需用 AppID+AppSecret 换取并缓存(有效期约 2 小时)。此模式要求
#   用户 48 小时内与公众号有过交互（刚发消息即满足）。


# -----------------------------------------------------------------------------
# 7. 启动
# -----------------------------------------------------------------------------
def main():
    mode = "安全模式(加密)" if _use_safe_mode() else "明文模式"
    print(f"模型: {MODEL}  后端: {BASE_URL or 'OpenAI 官方'}")
    print(f"微信公众号 Agent 启动: http://{HOST}:{PORT}{PATH}  [{mode}]")
    print("请在公众号后台「服务器配置」把 URL 指向本服务（需公网可达，如经内网穿透）。")
    server = ThreadingHTTPServer((HOST, PORT), WeChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.shutdown()


if __name__ == "__main__":
    main()
