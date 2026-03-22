"""
模型推理模块
支持加载基础模型（可选合并 LoRA adapter），提供流式对话生成。
"""

import gc
import threading
from pathlib import Path
from typing import Iterator

# 模块级单例，保持模型常驻内存
_model = None
_tokenizer = None
_current_base: str = ""
_current_adapter: str = ""


def is_loaded() -> bool:
    """返回当前是否有模型已加载"""
    return _model is not None


def get_loaded_info() -> str:
    """返回当前加载的模型简介"""
    if not is_loaded():
        return ""
    info = f"基础模型：`{Path(_current_base).name}`"
    if _current_adapter:
        info += f"  |  LoRA adapter：`{Path(_current_adapter).name}`"
    return info


def load_model(base_path: str, adapter_path: str = "") -> Iterator[str]:
    """
    Generator：加载基础模型 + 可选 LoRA adapter，逐步 yield 状态消息。
    adapter 使用 merge_and_unload 合并权重，推理更快。
    """
    global _model, _tokenizer, _current_base, _current_adapter

    base_path = base_path.strip()
    adapter_path = adapter_path.strip()

    if not base_path:
        yield "❌ 请选择基础模型路径"
        return

    if not Path(base_path).exists():
        yield f"❌ 路径不存在：{base_path}"
        return

    # 若已加载其他模型，先释放
    if _model is not None:
        yield "⏳ 释放旧模型内存..."
        _do_unload()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        yield "❌ 缺少依赖，请运行：pip install transformers torch"
        return

    # 自动选择设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    yield f"⏳ 正在加载 tokenizer...（设备：{device}）"
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            base_path, trust_remote_code=True
        )
    except Exception as e:
        yield f"❌ Tokenizer 加载失败：{e}"
        return

    yield f"⏳ 正在加载模型权重（首次约需 20~60 秒）..."
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as e:
        _model = None
        _tokenizer = None
        yield f"❌ 模型加载失败：{e}"
        return

    # 挂载 LoRA adapter
    if adapter_path and Path(adapter_path).exists():
        # 判断是否为有效 adapter 目录
        if not (Path(adapter_path) / "adapter_config.json").exists():
            yield f"⚠️ {adapter_path} 不是有效的 LoRA adapter，跳过挂载"
            _current_adapter = ""
        else:
            yield "⏳ 正在合并 LoRA adapter..."
            try:
                from peft import PeftModel
                _model = PeftModel.from_pretrained(_model, adapter_path)
                _model = _model.merge_and_unload()   # 合并权重，推理更快
                _current_adapter = adapter_path
                yield f"✅ LoRA adapter 合并完成：{Path(adapter_path).name}"
            except Exception as e:
                yield f"⚠️ LoRA adapter 挂载失败（{e}），使用纯基础模型"
                _current_adapter = ""
    else:
        _current_adapter = ""

    _model.eval()
    _current_base = base_path
    adapter_tag = f" + LoRA「{Path(_current_adapter).name}」" if _current_adapter else "（纯基础模型）"
    yield f"✅ 加载完成：{Path(base_path).name}{adapter_tag}  设备：{device}"


def _do_unload():
    """实际执行内存释放（内部调用）"""
    global _model, _tokenizer, _current_base, _current_adapter
    try:
        import torch
        del _model
        del _tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _model = None
    _tokenizer = None
    _current_base = ""
    _current_adapter = ""
    gc.collect()


def unload_model() -> str:
    """卸载模型，释放显存 / 内存"""
    if _model is None:
        return "⚠️ 当前没有已加载的模型"
    _do_unload()
    return "✅ 模型已卸载，内存已释放"


def scan_local_adapters() -> list[str]:
    """扫描 ./output 目录，返回包含 adapter_config.json 的目录路径"""
    output_dir = Path("./output")
    if not output_dir.exists():
        return []
    return [
        str(p.parent)
        for p in sorted(output_dir.rglob("adapter_config.json"))
    ]


def _history_to_chat_messages(history: list) -> list[dict]:
    """
    将 Gradio Chatbot 历史转为 LLM 用的 message 列表。
    支持 Gradio 6+ 的 [{role, content}, ...]，并兼容旧版 [user, assistant] 二元组。
    """
    out: list[dict] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in ("user", "assistant"):
                text = content if isinstance(content, str) else str(content or "")
                if text.strip():
                    out.append({"role": role, "content": text})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            u, a = item[0], item[1]
            if u:
                out.append({"role": "user", "content": str(u)})
            if a:
                out.append({"role": "assistant", "content": str(a)})
    return out


def chat_stream(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> Iterator[str]:
    """
    流式生成对话回复。
    history 为当前轮之前的对话：Gradio 6+ 使用 [{\"role\",\"content\"}, ...]；
    仍兼容旧版 [[user_msg, assistant_msg], ...]。
    每次 yield 当前累积的完整回复字符串。
    """
    if _model is None or _tokenizer is None:
        yield "⚠️ 请先在上方加载模型"
        return

    if not message.strip():
        yield ""
        return

    try:
        import torch
        from transformers import TextIteratorStreamer
    except ImportError:
        yield "❌ 缺少依赖 transformers"
        return

    # 构建 messages 列表
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(_history_to_chat_messages(history))
    messages.append({"role": "user", "content": message})

    # 应用对话模板
    try:
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = "\n".join(
            f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
            for m in messages
        ) + "\n助手: "

    inputs = _tokenizer(text, return_tensors="pt")
    device = next(_model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": int(max_new_tokens),
        "temperature": max(float(temperature), 1e-3),
        "do_sample": float(temperature) > 0.05,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "streamer": streamer,
    }

    # 在后台线程中运行生成，主线程流式读取
    thread = threading.Thread(target=_model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial
