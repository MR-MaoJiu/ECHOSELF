"""
模型推理模块
支持加载基础模型（可选合并 LoRA adapter），提供流式对话生成。
"""

import gc
import json
import re
import threading
from pathlib import Path
from typing import Iterator

# 模块级单例，保持模型常驻内存
_model = None
_tokenizer = None
_current_base: str = ""
_current_adapter: str = ""

# 已知模型架构 → 所需最低 transformers 版本
# 新模型发布时在此维护，升级提示会自动携带具体版本号
_ARCH_MIN_TRANSFORMERS: dict[str, str] = {
    "qwen3":          "4.51.0",
    "qwen3_moe":      "4.51.0",
    "qwen2":          "4.37.0",
    "qwen2_moe":      "4.40.0",
    "gemma3":         "4.49.0",
    "llama4":         "4.51.0",
    "mistral3":       "4.50.0",
    "deepseek_v3":    "4.45.0",
}


def _transformers_version() -> tuple[int, ...]:
    """返回当前 transformers 版本元组，导入失败时返回 (0,)。"""
    try:
        import transformers
        return tuple(int(x) for x in re.split(r"[.\-]", transformers.__version__)[:3] if x.isdigit())
    except Exception:
        return (0,)


def _version_tuple(ver_str: str) -> tuple[int, ...]:
    """将 '4.51.0' 转为 (4, 51, 0)。"""
    return tuple(int(x) for x in re.split(r"[.\-]", ver_str)[:3] if x.isdigit())


def _check_model_compatibility(model_path: str) -> str:
    """
    读取模型目录下的 config.json，检测 model_type 是否需要更高版本的 transformers。
    返回空字符串表示兼容，否则返回带升级命令的提示文字。
    跨平台：仅使用标准 Python + pip，Win / Mac / Linux 均适用。
    """
    config_file = Path(model_path) / "config.json"
    if not config_file.exists():
        return ""
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return ""

    model_type: str = config.get("model_type", "").lower()
    if not model_type:
        return ""

    min_ver_str = _ARCH_MIN_TRANSFORMERS.get(model_type, "")
    if not min_ver_str:
        return ""

    cur_ver = _transformers_version()
    min_ver = _version_tuple(min_ver_str)
    if cur_ver >= min_ver:
        return ""  # 版本足够，无需提示

    try:
        import transformers
        cur_ver_str = transformers.__version__
    except Exception:
        cur_ver_str = "未知"

    return (
        f"当前 transformers {cur_ver_str} 不支持 {model_type} 架构，"
        f"需要 >= {min_ver_str}。\n"
        "请升级后重启程序（任选其一）：\n"
        f'  pip install "transformers>={min_ver_str}"\n'
        f'  # 国内镜像：\n'
        f'  pip install "transformers>={min_ver_str}" '
        f'-i https://pypi.tuna.tsinghua.edu.cn/simple'
    )


def _arch_upgrade_hint(err: Exception) -> str:
    """
    判断异常是否为「架构不识别」，若是则返回升级提示；否则返回空串。
    作为加载失败后的二次诊断。
    """
    msg = str(err).lower()
    if "does not recognize this architecture" not in msg and "unknown model type" not in msg:
        return ""

    # 从错误信息中提取 model_type
    m = re.search(r"model type[:\s]+['\"]?(\w+)['\"]?", msg)
    detected = m.group(1) if m else ""
    min_ver_str = _ARCH_MIN_TRANSFORMERS.get(detected, "") if detected else ""

    try:
        import transformers
        cur_ver_str = transformers.__version__
    except Exception:
        cur_ver_str = "未知"

    pkg = f'"transformers>={min_ver_str}"' if min_ver_str else "--upgrade transformers"
    return (
        f"\n\n💡 当前 transformers {cur_ver_str} 不支持 {detected or '该'} 架构。\n"
        f"请升级后重启程序：\n"
        f"  pip install {pkg}\n"
        f"  # 国内镜像：\n"
        f"  pip install {pkg} -i https://pypi.tuna.tsinghua.edu.cn/simple"
    )


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

    # 权重文件预检：目录存在但缺少权重时立即提示重新下载，避免 transformers 抛出晦涩 OSError
    weight_patterns = [
        "pytorch_model.bin", "pytorch_model-*.bin",
        "model.safetensors",  "model-*.safetensors",
        "tf_model.h5", "model.ckpt.index", "flax_model.msgpack",
    ]
    _mp = Path(base_path)
    if not any(True for pat in weight_patterns for _ in _mp.glob(pat)):
        yield (
            f"❌ 模型目录缺少权重文件，请重新下载：\n   {base_path}\n\n"
            "未找到 pytorch_model.bin / model.safetensors 等权重文件。\n"
            "可能原因：下载未完成或中途中断。\n"
            "请在「⬇️ 模型下载」Tab 重新下载该模型。"
        )
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

    # 版本兼容性预检：读 config.json 判断 model_type 是否需要更高 transformers
    compat_hint = _check_model_compatibility(base_path)
    if compat_hint:
        yield f"❌ {compat_hint}"
        return

    # 自动选择设备（跨平台：MPS=Mac M 系列, CUDA=NVIDIA, 否则 CPU）
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
        hint = _arch_upgrade_hint(e)
        yield f"❌ Tokenizer 加载失败：{e}{hint}"
        return

    yield f"⏳ 正在加载模型权重（首次约需 20~60 秒）..."
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.bfloat16,   # torch_dtype 已弃用，改用 dtype
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as e:
        _model = None
        _tokenizer = None
        hint = _arch_upgrade_hint(e)
        yield f"❌ 模型加载失败：{e}{hint}"
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
