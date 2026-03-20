"""
聊天记录解析器
支持 WeFlow 导出的 JSON 格式（私聊）
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# 包含文本内容、可用于训练的消息类型
TRAINABLE_TYPES = {"文本消息", "引用消息"}

# 不含有效文本信息、需要跳过的消息类型
SKIP_TYPES = {
    "视频消息", "语音消息", "位置消息", "文件消息",
    "名片消息", "小程序消息", "转账消息", "红包消息",
    "系统消息", "撤回消息", "通话消息", "分享消息",
    "图片消息", "动画表情",
}


@dataclass
class ChatMessage:
    local_id: int
    create_time: int          # Unix 时间戳（秒）
    formatted_time: str
    msg_type: str
    content: str
    is_send: int              # 1 = 自己发出，0 = 对方发来
    sender_id: str            # 发送者账号 ID
    sender_name: str          # 发送者昵称
    quoted_content: Optional[str] = None  # 引用消息的原文内容


@dataclass
class ChatSession:
    """一段对话会话（对应一个联系人的所有消息）"""
    my_id: str                # 自己的账号 ID
    contact_name: str         # 对方昵称
    contact_id: str           # 对方账号 ID
    chat_type: str            # 对话类型（私聊 / 群聊）
    messages: list[ChatMessage] = field(default_factory=list)


def parse_json_file(file_path: str | Path, my_id: str = "") -> ChatSession:
    """
    解析单个聊天记录 JSON 文件

    Parameters
    ----------
    file_path : JSON 文件路径
    my_id : 自己的账号 ID，留空则自动从 isSend=1 的消息中推断
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    session_data = data.get("session", {})
    raw_messages = data.get("messages", [])

    # 自动推断自己的账号 ID
    if not my_id:
        for msg in raw_messages:
            if msg.get("isSend") == 1:
                my_id = msg.get("senderUsername", "")
                break

    session = ChatSession(
        my_id=my_id,
        contact_name=session_data.get("displayName") or session_data.get("nickname", ""),
        contact_id=session_data.get("wxid", ""),
        chat_type=session_data.get("type", "私聊"),
        messages=[],
    )

    for raw in raw_messages:
        msg = ChatMessage(
            local_id=raw.get("localId", 0),
            create_time=raw.get("createTime", 0),
            formatted_time=raw.get("formattedTime", ""),
            msg_type=raw.get("type", ""),
            content=raw.get("content", ""),
            is_send=raw.get("isSend", 0),
            sender_id=raw.get("senderUsername", ""),
            sender_name=raw.get("senderDisplayName", ""),
            quoted_content=raw.get("quotedContent"),
        )
        session.messages.append(msg)

    return session


def parse_multiple_files(file_paths: list[str | Path], my_id: str = "") -> list[ChatSession]:
    """解析多个聊天记录 JSON 文件"""
    sessions = []
    for fp in file_paths:
        try:
            session = parse_json_file(fp, my_id)
            sessions.append(session)
        except Exception as e:
            print(f"[parser] 跳过文件 {fp}，原因：{e}")
    return sessions


def get_message_type_stats(session: ChatSession) -> dict[str, int]:
    """统计消息类型分布，按数量降序排列"""
    stats: dict[str, int] = {}
    for msg in session.messages:
        stats[msg.msg_type] = stats.get(msg.msg_type, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
