"""
模型训练封装模块
通过子进程调用 LLaMA-Factory CLI 执行 LoRA 微调，支持日志流式回调和进程中断。
自动检测运行设备（Apple Silicon / CUDA / CPU），按需调整训练参数。
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

# 项目根目录（含 app.py、output/），子进程 cwd 固定为此，避免 Windows 下相对路径失效
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _abs_project_path(p: str) -> str:
    """将相对路径转为项目根目录下的绝对路径（Windows 下避免工作目录不一致导致找不到模型/数据）。"""
    pp = Path(p.strip())
    if not pp.is_absolute():
        pp = (PROJECT_ROOT / pp).resolve()
    return str(pp)


# ──────────────────────────────────────────────────────────
#  设备检测
# ──────────────────────────────────────────────────────────

def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _darwin_memory_gb() -> Optional[float]:
    """macOS 下读取物理内存容量（GB），失败返回 None。"""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0:
            return int(r.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return None


def _nvidia_smi_executable() -> Optional[str]:
    """返回 nvidia-smi 可执行文件路径（Windows 常见安装路径兜底）。"""
    w = shutil.which("nvidia-smi")
    if w:
        return w
    if platform.system() == "Windows":
        for p in (
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
        ):
            if Path(p).is_file():
                return p
    return None


def _nvidia_smi_gpu_info() -> tuple[bool, Optional[str], Optional[float]]:
    """
    不依赖 PyTorch，通过 nvidia-smi 检测 NVIDIA 显卡及显存（取第一块 GPU）。
    返回 (是否检测到, GPU 名称, 显存 GB)。
    """
    exe = _nvidia_smi_executable()
    if not exe:
        return False, None, None
    try:
        run_kw: dict = {
            "args": [
                exe,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            "capture_output": True,
            "text": True,
            "timeout": 10,
        }
        if platform.system() == "win32":
            run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        r = subprocess.run(**run_kw)
        if r.returncode != 0 or not (r.stdout or "").strip():
            return False, None, None
        line = (r.stdout or "").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            return False, None, None
        name = parts[0]
        mem_mib = float(parts[1])
        vram_gb = mem_mib / 1024.0
        return True, name, vram_gb
    except Exception:
        return False, None, None


def _format_cuda_setup_hint(
    gpu_name: str,
    *,
    torch_installed: bool = False,
    torch_version: Optional[str] = None,
    torch_cuda_build: Optional[str] = None,
) -> str:
    """检测到 NVIDIA 硬件但 PyTorch 未启用 CUDA 时，在 GUI 中展示的 Markdown 说明。"""
    extra = ""
    if torch_installed:
        tc = torch_cuda_build or "无（CPU 构建）"
        extra = (
            f"\n\n> **当前 PyTorch**：`{torch_version or '?'}`，CUDA 编译标记：`{tc}`\n"
        )
    return (
        "### 🖥️ 请启用 CUDA 以使用 NVIDIA 显卡训练\n\n"
        f"已检测到 **NVIDIA 显卡**：{gpu_name}，但当前 Python 中的 PyTorch **未启用 CUDA**，"
        "训练与推理会走 CPU。**按下列步骤安装带 CUDA 的 PyTorch 后，重启 EchoSelf：**\n\n"
        "1. **安装或更新显卡驱动**（与显卡型号匹配）："
        "[NVIDIA 驱动下载](https://www.nvidia.cn/drivers/)\n"
        "2. **安装 CUDA 版 PyTorch**：打开 [PyTorch 安装页](https://pytorch.org/get-started/locally/)，"
        "选择操作系统与 **CUDA** 版本，复制页面给出的 `pip` 命令。"
        "（示例：`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`，"
        "具体 `cu12xx` 以官网为准、需与驱动兼容。）\n"
        "3. 若曾安装过 CPU 版 PyTorch，建议先卸载再装：`pip uninstall torch torchvision torchaudio -y`，再执行官网命令。\n"
        "4. **验证**：终端执行 `python -c \"import torch; print(torch.cuda.is_available())\"`，应输出 `True`。\n"
        f"{extra}"
    )


def pick_recommended_for_budget(available_gb: float) -> tuple[list[str], str]:
    """
    按可用显存/统一内存（GB）估算可 LoRA 微调的预设模型。
    available_gb 会乘以安全系数后再与 MODEL_PRESETS 中的参考值比较。
    """
    # 略保守，避免在边界容量上 OOM
    ratio = 0.88
    threshold = max(available_gb * ratio, 1.0)
    names = [name for name, _, min_v in MODEL_PRESETS if min_v <= threshold]
    if not names:
        return (["Qwen2.5-0.5B-Instruct"], "Qwen2.5-0.5B-Instruct")
    by_min = {n: v for n, _, v in MODEL_PRESETS}
    default = max(names, key=lambda n: by_min[n])
    return (names, default)


def get_model_preset_choices(device_info: Optional[dict] = None) -> list[str]:
    """
    生成「模型下载」下拉里每一项的展示文案：参考显存 + 是否为本机推荐/可能超容量。
    """
    info = device_info or get_device_info()
    rec = set(info.get("recommended_models", []))
    budget = info.get("vram_gb")
    if budget is None:
        budget = info.get("memory_gb")
    if budget is None:
        budget = info.get("budget_gb")

    rows: list[str] = []
    for name, _tmpl, min_v in MODEL_PRESETS:
        parts = [f"约需 {min_v:.0f} GB 显存/统一内存"]
        if name in rec:
            parts.append("✅ 本机推荐")
        elif budget is not None and min_v > budget * 0.92:
            parts.append("⚠️ 可能超出本机容量")
        rows.append(f"{name}  ({' · '.join(parts)})")
    return rows


def get_device_info() -> dict:
    """
    返回当前设备信息，包括设备类型、推荐模型列表和注意事项。
    用于 GUI 展示和训练参数自动适配。

    额外字段：
    - cuda_ready: PyTorch 是否已成功启用 CUDA
    - nvidia_gpu_detected: 是否通过 nvidia-smi 检测到 NVIDIA 显卡
    - cuda_setup_hint: 有显卡但 CUDA 未就绪时的安装说明（Markdown），否则为空字符串
    - pytorch_cuda_build: torch.version.cuda（无则 None）
    """
    if is_apple_silicon():
        mem_gb = _darwin_memory_gb()
        budget = mem_gb if mem_gb is not None else 16.0
        rec_models, default_m = pick_recommended_for_budget(budget)
        note_mem = (
            f"检测到统一内存约 {mem_gb:.0f} GB，已据此推荐预设模型。"
            if mem_gb is not None
            else "未能读取内存容量，按常见配置推荐；请结合本机实际内存选择。"
        )
        return {
            "device": "Apple Silicon (MPS)",
            "icon": "🍎",
            "recommended_models": rec_models,
            "default_model": default_m,
            "memory_gb": mem_gb,
            "budget_gb": budget,
            "vram_gb": None,
            "notes": (
                "Apple Silicon 使用 MPS 后端训练，已自动禁用 fp16（改用 bf16）和 Flash Attention。\n"
                + note_mem
            ),
            "use_bf16": True,
            "use_fp16": False,
            "flash_attn": "disabled",
            "cuda_ready": False,
            "nvidia_gpu_detected": False,
            "nvidia_gpu_name": None,
            "cuda_setup_hint": "",
            "pytorch_cuda_build": None,
        }

    torch_mod = None
    try:
        import torch as torch_mod
    except ImportError:
        pass

    if torch_mod is not None and torch_mod.cuda.is_available():
        name = torch_mod.cuda.get_device_name(0)
        vram_gb = torch_mod.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        rec_models, default_m = pick_recommended_for_budget(vram_gb)
        tc = getattr(torch_mod.version, "cuda", None)
        return {
            "device": f"CUDA · {name} ({vram_gb:.0f} GB)",
            "icon": "🖥️",
            "recommended_models": rec_models,
            "default_model": default_m,
            "memory_gb": None,
            "budget_gb": vram_gb,
            "vram_gb": vram_gb,
            "notes": (
                f"检测到 GPU：{name}，显存 {vram_gb:.0f} GB，PyTorch 已启用 CUDA（{tc}），已据此推荐预设模型。"
            ),
            "use_bf16": False,
            "use_fp16": True,
            "flash_attn": "fa2",
            "cuda_ready": True,
            "nvidia_gpu_detected": True,
            "nvidia_gpu_name": name,
            "cuda_setup_hint": "",
            "pytorch_cuda_build": tc,
        }

    nv_ok, nv_name, nv_vram = _nvidia_smi_gpu_info()
    if nv_ok and nv_name is not None and nv_vram is not None:
        rec_models, default_m = pick_recommended_for_budget(nv_vram)
        tv = getattr(torch_mod, "__version__", None) if torch_mod is not None else None
        tc = getattr(torch_mod.version, "cuda", None) if torch_mod is not None else None
        hint = _format_cuda_setup_hint(
            nv_name,
            torch_installed=torch_mod is not None,
            torch_version=tv,
            torch_cuda_build=tc,
        )
        return {
            "device": f"NVIDIA 显卡已检测到（{nv_name}），但 PyTorch 未启用 CUDA",
            "icon": "🖥️",
            "recommended_models": rec_models,
            "default_model": default_m,
            "memory_gb": None,
            "budget_gb": nv_vram,
            "vram_gb": nv_vram,
            "notes": (
                "已通过 nvidia-smi 检测到 NVIDIA 显卡，但当前 Python 中的 PyTorch 未编译或未启用 CUDA，"
                "训练将使用 CPU。请查看界面顶部「CUDA 环境」说明安装带 CUDA 的 PyTorch 后重启。"
            ),
            "use_bf16": False,
            "use_fp16": False,
            "flash_attn": "disabled",
            "cuda_ready": False,
            "nvidia_gpu_detected": True,
            "nvidia_gpu_name": nv_name,
            "cuda_setup_hint": hint,
            "pytorch_cuda_build": tc,
        }

    return {
        "device": "CPU（不推荐训练）",
        "icon": "⚠️",
        "recommended_models": ["Qwen2.5-0.5B-Instruct"],
        "default_model": "Qwen2.5-0.5B-Instruct",
        "memory_gb": None,
        "budget_gb": 4.0,
        "vram_gb": None,
        "notes": "未检测到 GPU/MPS，CPU 训练速度极慢，仅供调试使用。",
        "use_bf16": False,
        "use_fp16": False,
        "flash_attn": "disabled",
        "cuda_ready": False,
        "nvidia_gpu_detected": False,
        "nvidia_gpu_name": None,
        "cuda_setup_hint": "",
        "pytorch_cuda_build": getattr(torch_mod.version, "cuda", None) if torch_mod is not None else None,
    }


# ──────────────────────────────────────────────────────────
#  训练配置
# ──────────────────────────────────────────────────────────

# 模型预设：(显示名称, 模板名称, LoRA 微调参考显存/统一内存 GB，近似值)
# Qwen3 需 transformers≥4.51，LLaMA-Factory 中 template 一般为 qwen3；训练前请升级环境。
MODEL_PRESETS: list[tuple[str, str, float]] = [
    # Qwen3（新一代，template: qwen3）
    ("Qwen3-0.6B", "qwen3", 2.0),
    ("Qwen3-1.7B", "qwen3", 4.0),
    ("Qwen3-4B-Instruct-2507", "qwen3", 8.0),
    ("Qwen3-8B", "qwen3", 16.0),
    ("Qwen3-14B", "qwen3", 28.0),
    ("Qwen3-32B", "qwen3", 48.0),
    ("Qwen3-Coder-30B-A3B-Instruct", "qwen3", 28.0),
    # Qwen2 / Qwen2.5
    ("Qwen2.5-0.5B-Instruct", "qwen", 2.0),
    ("Qwen2-1.5B-Instruct", "qwen", 4.0),
    ("Qwen2.5-1.5B-Instruct", "qwen", 4.0),
    ("Qwen2-7B-Instruct", "qwen", 16.0),
    ("Qwen2.5-3B-Instruct", "qwen", 8.0),
    ("Qwen2.5-7B-Instruct", "qwen", 16.0),
    ("Qwen2.5-14B-Instruct", "qwen", 32.0),
    ("Qwen2.5-32B-Instruct", "qwen", 48.0),
    ("Qwen2.5-72B-Instruct", "qwen", 96.0),
    ("Qwen2.5-Math-1.5B-Instruct", "qwen", 4.0),
    ("Qwen2.5-Math-7B-Instruct", "qwen", 16.0),
    ("Qwen2.5-Coder-1.5B-Instruct", "qwen", 4.0),
    ("Qwen2.5-Coder-7B-Instruct", "qwen", 16.0),
    # Llama 3.x / 3.3
    ("Llama-3.2-1B-Instruct", "llama3", 3.0),
    ("Llama-3.2-3B-Instruct", "llama3", 8.0),
    ("Llama-3.1-8B-Instruct", "llama3", 16.0),
    ("Llama-3.1-70B-Instruct", "llama3", 96.0),
    ("Llama-3.3-8B-Instruct", "llama3", 16.0),
    ("Llama-3.3-70B-Instruct", "llama3", 96.0),
    # 轻量 / 通用小模型
    ("TinyLlama-1.1B-Chat-v1.0", "tinyllama", 2.0),
    ("SmolLM2-1.7B-Instruct", "default", 4.0),
    # Mistral / Gemma
    ("Mistral-7B-Instruct-v0.3", "mistral", 16.0),
    ("Mistral-Small-24B-Instruct-2501", "mistral", 40.0),
    ("gemma-2-2b-it", "gemma", 6.0),
    ("gemma-2-9b-it", "gemma", 18.0),
    ("gemma-2-27b-it", "gemma", 36.0),
    # 国产 / 多系列
    ("Yi-1.5-6B-Chat", "yi", 10.0),
    ("Yi-1.5-9B-Chat", "yi", 14.0),
    ("InternLM2-Chat-1.8B", "intern2", 4.0),
    ("InternLM2-Chat-7B", "intern2", 16.0),
    ("InternLM3-8B-Instruct", "intern3", 16.0),
    ("ChatGLM3-6B", "chatglm3", 12.0),
    ("DeepSeek-Coder-1.3B-Instruct", "deepseekcoder", 3.0),
    ("DeepSeek-Coder-6.7B-Instruct", "deepseekcoder", 14.0),
    ("DeepSeek-R1-Distill-Qwen-7B", "qwen", 16.0),
    ("DeepSeek-R1-Distill-Llama-8B", "llama3", 16.0),
    # Phi
    ("Phi-3-mini-4k-instruct", "phi", 8.0),
    ("Phi-3-medium-4k-instruct", "phi", 28.0),
    ("Phi-3.5-mini-instruct", "phi", 8.0),
    ("Phi-4-mini-instruct", "phi", 8.0),
]

# 模板映射（模型名 → 模板名）
_MODEL_TEMPLATE_MAP = {preset[0]: preset[1] for preset in MODEL_PRESETS}


def get_template_for_model(model_name_or_path: str) -> str:
    """根据模型路径/名称自动推断对应的对话模板"""
    name = Path(model_name_or_path).name
    for model_key, template in _MODEL_TEMPLATE_MAP.items():
        if model_key.lower() in name.lower():
            return template
    # 按关键词兜底匹配（与 LLaMA-Factory 模板名对齐）
    name_lower = name.lower()
    if "tinyllama" in name_lower:
        return "tinyllama"
    if "r1-distill-qwen" in name_lower or "r1_distill_qwen" in name_lower:
        return "qwen"
    if "r1-distill-llama" in name_lower or "r1_distill_llama" in name_lower:
        return "llama3"
    if "deepseek" in name_lower and "coder" in name_lower:
        return "deepseekcoder"
    if "deepseek" in name_lower:
        return "deepseekcoder"
    if "internlm3" in name_lower:
        return "intern3"
    if "internlm" in name_lower:
        return "intern2"
    if "chatglm" in name_lower:
        return "chatglm3"
    if "mistral" in name_lower or "mixtral" in name_lower:
        return "mistral"
    if "gemma" in name_lower:
        return "gemma"
    if "yi-" in name_lower or "yi_1.5" in name_lower or "/yi-" in name_lower:
        return "yi"
    if "qwen3" in name_lower:
        return "qwen3"
    if "qwen2" in name_lower:
        return "qwen"
    if "qwen" in name_lower:
        return "qwen"
    if "llama" in name_lower:
        return "llama3"
    if "phi" in name_lower:
        return "phi"
    return "default"


@dataclass
class TrainConfig:
    """训练参数配置（含设备自动适配逻辑）"""
    # 模型
    model_name_or_path: str = "./models/Qwen2.5-1.5B-Instruct"
    template: str = "qwen"

    # 数据
    dataset_path: str = "./output/sft_data.json"
    dataset_dir: str = "./output"
    dataset_name: str = "echoself_sft"
    output_dir: str = "./output/model"

    # 系统提示词
    default_system: str = "请你扮演一个真实的人，用自然的方式进行对话。"

    # LoRA 参数
    finetuning_type: str = "lora"
    lora_target: str = "q_proj,v_proj"
    lora_rank: int = 8
    lora_dropout: float = 0.1

    # 训练超参
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    cutoff_len: int = 512
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # 精度（由设备自动适配，请勿手动设置不兼容的组合）
    fp16: bool = False
    bf16: bool = False
    flash_attn: str = "disabled"

    # 日志 & 保存
    logging_steps: int = 10
    save_steps: int = 100
    plot_loss: bool = True
    overwrite_cache: bool = True
    overwrite_output_dir: bool = True

    # 续训：指定 checkpoint 目录路径（如 ./output/model/checkpoint-100），None 表示全新训练
    resume_from_checkpoint: Optional[str] = None

    def auto_adjust(self) -> "TrainConfig":
        """根据当前设备自动调整精度和 Flash Attention 设置"""
        info = get_device_info()
        self.fp16 = info["use_fp16"]
        self.bf16 = info["use_bf16"]
        self.flash_attn = info["flash_attn"]
        return self


def _resolve_train_config(config: TrainConfig) -> TrainConfig:
    """训练前统一为绝对路径，并修正 dataset_dir。"""
    ds = _abs_project_path(config.dataset_path)
    out = _abs_project_path(config.output_dir)
    model = _abs_project_path(config.model_name_or_path)
    resume = config.resume_from_checkpoint
    if resume:
        rp = Path(resume.strip())
        if not rp.is_absolute():
            rp = (PROJECT_ROOT / rp).resolve()
        resume = str(rp)
    else:
        resume = None
    return replace(
        config,
        model_name_or_path=model,
        dataset_path=ds,
        dataset_dir=str(Path(ds).parent),
        output_dir=out,
        resume_from_checkpoint=resume,
    )


def _validate_training_inputs(cfg: TrainConfig, log_callback: Callable[[str], None]) -> bool:
    """启动子进程前校验模型、数据是否存在且非空。"""
    if not Path(cfg.model_name_or_path).exists():
        log_callback(f"❌ 基础模型路径不存在：\n   {cfg.model_name_or_path}")
        return False
    if not Path(cfg.dataset_path).exists():
        log_callback(f"❌ 训练数据文件不存在：\n   {cfg.dataset_path}")
        return False
    try:
        with open(cfg.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            log_callback("❌ 训练数据为空（JSON 数组长度为 0），请先在「数据处理」中生成有效样本。")
            return False
    except Exception as e:
        log_callback(f"❌ 无法读取训练数据文件：{e}")
        return False
    if cfg.resume_from_checkpoint and not Path(cfg.resume_from_checkpoint).exists():
        log_callback(f"❌ 续训 checkpoint 路径不存在：\n   {cfg.resume_from_checkpoint}")
        return False
    return True


def has_training_artifacts(output_dir: str) -> bool:
    """输出目录下是否存在 LoRA adapter 或 HuggingFace 式 checkpoint 子目录。"""
    p = Path(_abs_project_path(output_dir))
    if not p.is_dir():
        return False
    if (p / "adapter_config.json").exists():
        return True
    for cp in p.glob("checkpoint-*"):
        if cp.is_dir():
            return True
    return False


# ──────────────────────────────────────────────────────────
#  LLaMA-Factory 接口
# ──────────────────────────────────────────────────────────

def check_llamafactory() -> tuple[bool, str]:
    """
    检测 LLaMA-Factory 是否已安装在「当前 EchoSelf 使用的 Python」中。
    始终用 python -m 方式检测，避免 Windows 上 PATH 里的 llamafactory-cli 指向其它 Python 导致假阳性/训练不生效。
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "llamafactory.cli", "--help"],
            capture_output=True,
            timeout=20,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            return True, f"{sys.executable} -m llamafactory.cli"
    except Exception:
        pass
    return False, ""


def _detect_data_format(dataset_path: str) -> str:
    """
    读取数据文件第一条记录，自动判断是 alpaca 还是 sharegpt 格式。
    sharegpt: 顶层字段含 'conversations'
    alpaca:   顶层字段含 'instruction'
    """
    try:
        with open(dataset_path, encoding="utf-8") as f:
            import json as _json
            data = _json.load(f)
        first = data[0] if data else {}
        if "conversations" in first:
            return "sharegpt"
    except Exception:
        pass
    return "alpaca"


def _build_dataset_info(config: TrainConfig) -> dict:
    fmt = _detect_data_format(config.dataset_path)
    if fmt == "sharegpt":
        # ShareGPT 格式：conversations 列表，每条含 from/value
        entry = {
            "file_name": Path(config.dataset_path).name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    else:
        # Alpaca 格式：instruction / output / system
        entry = {
            "file_name": Path(config.dataset_path).name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "response": "output",
                "system": "system",
            },
        }
    return {config.dataset_name: entry}


def _build_train_args(config: TrainConfig) -> dict:
    """构建传递给 LLaMA-Factory 的参数字典（排除冲突的精度标志）"""
    args: dict = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": config.model_name_or_path,
        "template": config.template,
        "dataset": config.dataset_name,
        "dataset_dir": config.dataset_dir,
        "output_dir": config.output_dir,
        "finetuning_type": config.finetuning_type,
        "lora_target": config.lora_target,
        "lora_rank": config.lora_rank,
        "lora_dropout": config.lora_dropout,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "cutoff_len": config.cutoff_len,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "flash_attn": config.flash_attn,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "plot_loss": config.plot_loss,
        "overwrite_cache": config.overwrite_cache,
        # 续训时不覆盖输出目录，以保留已有的 adapter 和 checkpoint
        "overwrite_output_dir": False if config.resume_from_checkpoint else config.overwrite_output_dir,
        "default_system": config.default_system,
        "trust_remote_code": True,
        # 禁用 wandb / tensorboard 等实验追踪，避免需要登录 API key
        "report_to": "none",
    }
    # 续训参数：传入 checkpoint 目录路径
    if config.resume_from_checkpoint:
        args["resume_from_checkpoint"] = config.resume_from_checkpoint
    # fp16 和 bf16 互斥，只传入 True 的那个
    if config.bf16:
        args["bf16"] = True
    elif config.fp16:
        args["fp16"] = True
    return args


def prepare_train_files(config: TrainConfig) -> None:
    """写入 dataset_info.json 和训练参数快照"""
    dataset_dir = Path(config.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(_build_dataset_info(config), f, ensure_ascii=False, indent=2)

    with open(dataset_dir / "train_args_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(_build_train_args(config), f, ensure_ascii=False, indent=2)


def get_train_command(config: TrainConfig) -> str:
    """生成等价的 bash 训练命令"""
    args = _build_train_args(config)
    lines = ["llamafactory-cli train \\"]
    for k, v in args.items():
        val = str(v).lower() if isinstance(v, bool) else str(v)
        lines.append(f"  --{k} {val} \\")
    lines[-1] = lines[-1].rstrip(" \\")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
#  训练进程管理
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
#  模型下载
# ──────────────────────────────────────────────────────────

# 各预设模型对应的下载 ID：(ModelScope ID, HuggingFace ID)
MODEL_DOWNLOAD_IDS: dict[str, tuple[str, str]] = {
    # Qwen3
    "Qwen3-0.6B": ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
    "Qwen3-1.7B": ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
    "Qwen3-4B-Instruct-2507": ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
    "Qwen3-14B": ("Qwen/Qwen3-14B", "Qwen/Qwen3-14B"),
    "Qwen3-32B": ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B"),
    "Qwen3-Coder-30B-A3B-Instruct": (
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    ),
    # Qwen2 / Qwen2.5
    "Qwen2.5-0.5B-Instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    "Qwen2-1.5B-Instruct": ("Qwen/Qwen2-1.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct"),
    "Qwen2.5-1.5B-Instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    "Qwen2-7B-Instruct": ("Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-7B-Instruct"),
    "Qwen2.5-3B-Instruct": ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
    "Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    "Qwen2.5-14B-Instruct": ("Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct"),
    "Qwen2.5-32B-Instruct": ("Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Instruct"),
    "Qwen2.5-72B-Instruct": ("Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct"),
    "Qwen2.5-Math-1.5B-Instruct": ("Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
    "Qwen2.5-Math-7B-Instruct": ("Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-Math-7B-Instruct"),
    "Qwen2.5-Coder-1.5B-Instruct": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    "Qwen2.5-Coder-7B-Instruct": ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    # Llama 3.x / 3.3
    "Llama-3.2-1B-Instruct": ("LLM-Research/Meta-Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    "Llama-3.2-3B-Instruct": ("LLM-Research/Meta-Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    "Llama-3.1-8B-Instruct": ("LLM-Research/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "Llama-3.1-70B-Instruct": ("LLM-Research/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
    "Llama-3.3-8B-Instruct": ("LLM-Research/Meta-Llama-3.3-8B-Instruct", "meta-llama/Meta-Llama-3.3-8B-Instruct"),
    "Llama-3.3-70B-Instruct": ("LLM-Research/Meta-Llama-3.3-70B-Instruct", "meta-llama/Meta-Llama-3.3-70B-Instruct"),
    "TinyLlama-1.1B-Chat-v1.0": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    "SmolLM2-1.7B-Instruct": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    "Mistral-7B-Instruct-v0.3": ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    "Mistral-Small-24B-Instruct-2501": (
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501",
    ),
    "gemma-2-2b-it": ("google/gemma-2-2b-it", "google/gemma-2-2b-it"),
    "gemma-2-9b-it": ("google/gemma-2-9b-it", "google/gemma-2-9b-it"),
    "gemma-2-27b-it": ("google/gemma-2-27b-it", "google/gemma-2-27b-it"),
    "Yi-1.5-6B-Chat": ("01-ai/Yi-1.5-6B-Chat", "01-ai/Yi-1.5-6B-Chat"),
    "Yi-1.5-9B-Chat": ("01-ai/Yi-1.5-9B-Chat", "01-ai/Yi-1.5-9B-Chat"),
    "InternLM2-Chat-1.8B": ("Shanghai_AI_Laboratory/internlm2-chat-1_8b", "internlm/internlm2-chat-1_8b"),
    "InternLM2-Chat-7B": ("Shanghai_AI_Laboratory/internlm2-chat-7b", "internlm/internlm2-chat-7b"),
    "InternLM3-8B-Instruct": (
        "Shanghai_AI_Laboratory/internlm3-8b-instruct",
        "internlm/internlm3-8b-instruct",
    ),
    "ChatGLM3-6B": ("ZhipuAI/chatglm3-6b", "THUDM/chatglm3-6b"),
    "DeepSeek-Coder-1.3B-Instruct": ("deepseek-ai/deepseek-coder-1.3b-instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
    "DeepSeek-Coder-6.7B-Instruct": ("deepseek-ai/deepseek-coder-6.7b-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"),
    "DeepSeek-R1-Distill-Qwen-7B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    "DeepSeek-R1-Distill-Llama-8B": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    "Phi-3-mini-4k-instruct": ("microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-4k-instruct"),
    "Phi-3-medium-4k-instruct": ("microsoft/Phi-3-medium-4k-instruct", "microsoft/Phi-3-medium-4k-instruct"),
    "Phi-3.5-mini-instruct": ("microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-mini-instruct"),
    "Phi-4-mini-instruct": ("microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-instruct"),
}


def check_modelscope() -> bool:
    """检测 modelscope 是否已安装"""
    import importlib.util
    return importlib.util.find_spec("modelscope") is not None


def check_huggingface_hub() -> bool:
    """检测 huggingface_hub 是否已安装"""
    import importlib.util
    return importlib.util.find_spec("huggingface_hub") is not None


class DownloadProcess:
    """封装模型下载子进程，支持流式日志输出和中途停止"""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stopped = False

    def start(
        self,
        source: str,
        model_id: str,
        local_dir: str,
        log_callback: Callable[[str], None],
        done_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """
        启动下载子进程。
        source: "modelscope" 或 "huggingface"
        """
        if source == "modelscope":
            cmd = [
                sys.executable, "-m", "modelscope", "download",
                "--model", model_id,
                "--local_dir", local_dir,
            ]
        else:
            # 优先使用 huggingface-cli，否则通过 huggingface_hub Python 模块
            hf_cli = shutil.which("huggingface-cli")
            if hf_cli:
                cmd = [hf_cli, "download", model_id, "--local-dir", local_dir]
            else:
                cmd = [
                    sys.executable, "-c",
                    (
                        f"from huggingface_hub import snapshot_download; "
                        f"snapshot_download('{model_id}', local_dir='{local_dir}')"
                    ),
                ]

        self._stopped = False
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        popen_kw = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        if platform.system() == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            self._process = subprocess.Popen(cmd, **popen_kw)
        except FileNotFoundError as e:
            log_callback(f"❌ 启动失败：{e}")
            return False

        def _stream():
            assert self._process and self._process.stdout
            for line in self._process.stdout:
                if self._stopped:
                    break
                log_callback(line.rstrip())
            self._process.wait()
            if done_callback:
                done_callback(self._process.returncode)

        threading.Thread(target=_stream, daemon=True).start()
        return True

    def stop(self):
        self._stopped = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None


def scan_checkpoints(output_dir: str) -> list[str]:
    """
    扫描训练输出目录中的 checkpoint 子目录，按步数从大到小排序返回。
    LLaMA-Factory 保存的 checkpoint 目录名格式为 checkpoint-{step}。
    """
    base = Path(output_dir)
    if not base.exists():
        return []
    checkpoints = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    return [str(c) for c in checkpoints]


class TrainingProcess:
    """封装 LLaMA-Factory 训练子进程，支持流式日志输出和中途停止"""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stopped = False

    def start(
        self,
        config: TrainConfig,
        log_callback: Callable[[str], None],
        done_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        # 防止重复启动：若旧进程仍在运行直接拒绝
        if self.is_running:
            log_callback("⚠️ 训练进程仍在运行中，请先点击「⏹️ 停止训练」后再重新开始。")
            return False

        ok, _cli_path = check_llamafactory()
        if not ok:
            log_callback(
                "❌ 当前 Python 环境中未找到 LLaMA-Factory。请在「用于启动 EchoSelf 的同一环境」中安装：\n"
                f"   `{sys.executable} -m pip install llamafactory`"
            )
            return False

        # 自动适配设备参数 + 绝对路径（避免 Windows 下相对路径解析错误导致秒退）
        config.auto_adjust()
        cfg = _resolve_train_config(config)
        if not _validate_training_inputs(cfg, log_callback):
            return False

        prepare_train_files(cfg)

        args = _build_train_args(cfg)
        # 必须用与 GUI 相同的解释器启动，Windows 下勿依赖 PATH 中的 llamafactory-cli
        cmd = [sys.executable, "-m", "llamafactory.cli", "train"]
        for k, v in args.items():
            cmd += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]

        log_callback(f"📂 训练工作目录：{PROJECT_ROOT}")
        log_callback("▶ 启动 LLaMA-Factory（首行日志若迟迟不出现，请稍等模型加载）…")

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # 禁用 wandb 自动登录，避免没有 API key 时训练崩溃
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
        }
        self._stopped = False

        # bufsize=1 行缓冲，便于尽快看到日志；stdin=DEVNULL 避免 Windows 上意外阻塞；
        # 不使用 CREATE_NO_WINDOW，部分环境下该标志会导致子进程异常秒退且无输出。
        popen_kw: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "env": env,
            "cwd": str(PROJECT_ROOT),
            "bufsize": 1,
        }

        try:
            self._process = subprocess.Popen(cmd, **popen_kw)
        except FileNotFoundError as e:
            log_callback(f"❌ 启动失败：{e}")
            return False

        def _stream():
            assert self._process and self._process.stdout
            line_count = 0
            for line in self._process.stdout:
                if self._stopped:
                    break
                line_count += 1
                log_callback(line.rstrip())
            rc = self._process.wait()
            if line_count == 0:
                log_callback(
                    "⚠️ 子进程未向标准输出打印任何内容即已结束。"
                    "若下方显示成功但无权重文件，请在本机终端手动运行：\n"
                    f"   cd /d \"{PROJECT_ROOT}\"\n"
                    f"   \"{sys.executable}\" -m llamafactory.cli train ...\n"
                    "或检查是否缺少依赖、CUDA 与模型路径是否正确。"
                )
            if done_callback:
                done_callback(rc)

        threading.Thread(target=_stream, daemon=True).start()
        return True

    def stop(self):
        self._stopped = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
