"""
数据预处理模块
职责：消息过滤 → 隐私脱敏 → 连续消息合并 → QA 对构建 → 训练格式输出
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.parser import ChatMessage, ChatSession


# ──────────────────────────────────────────────────────────
#  隐私脱敏（PII Removal）
# ──────────────────────────────────────────────────────────

# 正则规则：(匹配模式, 替换文本)
_PII_RULES: list[tuple[str, str]] = [
    (r"(?<!\d)1[3-9]\d{9}(?!\d)",                                                  "[手机号]"),
    (r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)",                                       "[座机号]"),
    (r"[\w.+\-]+@[\w\-]+\.[\w.\-]+",                                               "[邮箱]"),
    (r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)", "[身份证]"),
    (r"(?<!\d)[4-9]\d{15,18}(?!\d)",                                               "[银行卡]"),
    (r"QQ[号：:\s]*([1-9]\d{4,10})",                                               "QQ[QQ号]"),
    (r"(?:微信[号：:\s]*)([a-zA-Z][a-zA-Z0-9_\-]{5,19})",                          "微信[账号]"),
    (r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)",                                     "[IP地址]"),
    (r"(?:住在|地址[是为：:]\s*|家在)([^\s，。！？,!?]{4,20})",                      "[地址]"),
]


def remove_pii(text: str, blocked_words: list[str] | None = None) -> tuple[str, bool]:
    """
    对文本执行隐私脱敏。

    Returns
    -------
    (脱敏后文本, 是否触发脱敏)
    返回空字符串表示整条消息应被丢弃（命中禁用词）
    """
    # 禁用词：命中则整条消息丢弃
    if blocked_words:
        for word in blocked_words:
            if word and word in text:
                return "", True

    has_pii = False
    for pattern, replacement in _PII_RULES:
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            has_pii = True
            text = new_text

    return text, has_pii


# ──────────────────────────────────────────────────────────
#  消息过滤
# ──────────────────────────────────────────────────────────

_KEEP_TYPES = {"文本消息", "引用消息"}

# 纯占位内容，无训练价值
_PLACEHOLDER_RE = re.compile(
    r"^\s*(\[视频\]|\[语音\]|\[图片\]|\[文件\]|\[位置\]|\[表情\]|\[动画表情\])\s*$"
)


def _is_valid_content(content: str, min_len: int, max_len: int) -> bool:
    content = content.strip()
    if not content or len(content) < min_len or len(content) > max_len:
        return False
    if _PLACEHOLDER_RE.match(content):
        return False
    # 过滤纯方括号表情内容，如"[旺柴][旺柴]"
    if not re.sub(r"\[[^\]]{1,10}\]", "", content).strip():
        return False
    return True


# ──────────────────────────────────────────────────────────
#  QA 对数据结构
# ──────────────────────────────────────────────────────────

@dataclass
class QAPair:
    instruction: str   # 对方发来的内容（输入）
    output: str        # 自己的回复（输出）
    time: str = ""
    system: str = ""


@dataclass
class ProcessResult:
    qa_pairs: list[QAPair] = field(default_factory=list)
    total_messages: int = 0
    kept_messages: int = 0
    pii_removed: int = 0
    blocked_removed: int = 0
    skipped_type: int = 0
    skipped_length: int = 0


# ──────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────

def build_qa_pairs(
    session: ChatSession,
    time_window_minutes: int = 5,
    single_combine_window_minutes: int = 2,
    min_msg_len: int = 2,
    max_msg_len: int = 500,
    blocked_words: list[str] | None = None,
    enable_pii_removal: bool = True,
    system_prompt: str = "请你扮演一个真实的人，用自然的方式进行对话。",
) -> ProcessResult:
    """
    将一个会话的消息列表转换为 QA 训练对。

    流程：类型过滤 → PII 脱敏 → 连续消息合并 → 时间窗口 QA 匹配
    """
    result = ProcessResult(total_messages=len(session.messages))
    blocked_words = blocked_words or []

    # 1. 类型 & 长度过滤
    filtered: list[ChatMessage] = []
    for msg in session.messages:
        if msg.msg_type not in _KEEP_TYPES:
            result.skipped_type += 1
            continue
        if not _is_valid_content(msg.content, min_msg_len, max_msg_len):
            result.skipped_length += 1
            continue
        filtered.append(msg)

    # 2. PII 脱敏 & 禁用词过滤
    cleaned: list[ChatMessage] = []
    for msg in filtered:
        if enable_pii_removal:
            new_content, had_pii = remove_pii(msg.content, blocked_words)
        else:
            new_content, had_pii = msg.content, False

        if not new_content:
            result.blocked_removed += 1
            continue
        if had_pii:
            result.pii_removed += 1

        msg.content = new_content
        cleaned.append(msg)

    result.kept_messages = len(cleaned)

    # 3. 合并同人连续消息
    combined = _combine_consecutive(cleaned, single_combine_window_minutes * 60)

    # 4. 时间窗口 QA 匹配
    result.qa_pairs = _match_qa_pairs(combined, time_window_minutes * 60, system_prompt)

    return result


def _combine_consecutive(messages: list[ChatMessage], window_sec: int) -> list[ChatMessage]:
    """将连续同一发送方、时间窗口内的消息合并为一条"""
    if not messages:
        return []

    result: list[ChatMessage] = []
    cur = messages[0]

    for msg in messages[1:]:
        same_sender = msg.is_send == cur.is_send
        in_window = (msg.create_time - cur.create_time) <= window_sec
        if same_sender and in_window:
            cur = ChatMessage(
                local_id=cur.local_id,
                create_time=msg.create_time,
                formatted_time=msg.formatted_time,
                msg_type=cur.msg_type,
                content=cur.content + "\n" + msg.content,
                is_send=cur.is_send,
                sender_id=cur.sender_id,
                sender_name=cur.sender_name,
            )
        else:
            result.append(cur)
            cur = msg

    result.append(cur)
    return result


def _match_qa_pairs(
    messages: list[ChatMessage],
    window_sec: int,
    system_prompt: str,
) -> list[QAPair]:
    """
    时间窗口 QA 匹配策略：
    找到对方消息（is_send=0），向后寻找自己的回复（is_send=1），
    两者时间差在 window_sec 内即构成有效 QA 对。
    """
    pairs: list[QAPair] = []
    i = 0

    while i < len(messages) - 1:
        msg = messages[i]

        if msg.is_send != 0:
            i += 1
            continue

        # 查找紧随其后的自己的回复
        j = i + 1
        while j < len(messages) and messages[j].is_send == 0:
            if (messages[j].create_time - messages[j - 1].create_time) > window_sec:
                break
            j += 1

        if j >= len(messages) or messages[j].is_send != 1:
            i = j if j < len(messages) else i + 1
            continue

        my_msg = messages[j]
        time_diff = my_msg.create_time - msg.create_time

        if time_diff <= window_sec:
            instruction = msg.content.strip()
            output = my_msg.content.strip()
            if instruction and output:
                pairs.append(QAPair(
                    instruction=instruction,
                    output=output,
                    time=msg.formatted_time,
                    system=system_prompt,
                ))

        i = j + 1

    return pairs


# ──────────────────────────────────────────────────────────
#  训练数据格式输出
# ──────────────────────────────────────────────────────────

def to_alpaca_format(qa_pairs: list[QAPair]) -> list[dict]:
    """Alpaca 格式：instruction / input / output / system"""
    return [
        {
            "instruction": p.instruction,
            "input": "",
            "output": p.output,
            "system": p.system,
        }
        for p in qa_pairs
    ]


def to_sharegpt_format(qa_pairs: list[QAPair]) -> list[dict]:
    """ShareGPT 格式：conversations 多轮对话"""
    return [
        {
            "conversations": [
                {"from": "human", "value": p.instruction},
                {"from": "gpt", "value": p.output},
            ],
            "system": p.system,
        }
        for p in qa_pairs
    ]


def save_dataset(
    qa_pairs: list[QAPair],
    output_path: str | Path,
    fmt: str = "alpaca",
) -> int:
    """将 QA 对保存为 JSON 文件，返回保存条数"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = to_alpaca_format(qa_pairs) if fmt == "alpaca" else to_sharegpt_format(qa_pairs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(data)
