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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────
#  设备检测
# ──────────────────────────────────────────────────────────

def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_device_info() -> dict:
    """
    返回当前设备信息，包括设备类型、推荐模型列表和注意事项。
    用于 GUI 展示和训练参数自动适配。
    """
    if is_apple_silicon():
        return {
            "device": "Apple Silicon (MPS)",
            "icon": "🍎",
            "recommended_models": [
                "Qwen2.5-0.5B-Instruct",
                "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-3B-Instruct",
            ],
            "default_model": "Qwen2.5-1.5B-Instruct",
            "notes": (
                "M 系列芯片使用 MPS 后端训练，已自动禁用 fp16（改用 bf16）和 Flash Attention。\n"
                "16GB 内存推荐使用 1.5B 或 3B 模型，LoRA 训练内存占用约 6~10 GB。"
            ),
            "use_bf16": True,
            "use_fp16": False,
            "flash_attn": "disabled",
        }

    # 尝试检测 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return {
                "device": f"CUDA · {name} ({vram_gb:.0f} GB)",
                "icon": "🖥️",
                "recommended_models": [
                    "Qwen2.5-1.5B-Instruct",
                    "Qwen2.5-3B-Instruct",
                    "Qwen2.5-7B-Instruct",
                ],
                "default_model": "Qwen2.5-7B-Instruct",
                "notes": f"检测到 GPU：{name}，显存 {vram_gb:.0f} GB",
                "use_bf16": False,
                "use_fp16": True,
                "flash_attn": "fa2",
            }
    except ImportError:
        pass

    return {
        "device": "CPU（不推荐训练）",
        "icon": "⚠️",
        "recommended_models": ["Qwen2.5-0.5B-Instruct"],
        "default_model": "Qwen2.5-0.5B-Instruct",
        "notes": "未检测到 GPU/MPS，CPU 训练速度极慢，仅供调试使用。",
        "use_bf16": False,
        "use_fp16": False,
        "flash_attn": "disabled",
    }


# ──────────────────────────────────────────────────────────
#  训练配置
# ──────────────────────────────────────────────────────────

# 模型预设：(显示名称, 模板名称, 显存需求提示)
MODEL_PRESETS: list[tuple[str, str, str]] = [
    ("Qwen2.5-0.5B-Instruct", "qwen",    "~2 GB · Mac M4 ✅"),
    ("Qwen2.5-1.5B-Instruct", "qwen",    "~4 GB · Mac M4 ✅"),
    ("Qwen2.5-3B-Instruct",   "qwen",    "~8 GB · Mac M4 ✅"),
    ("Qwen2.5-7B-Instruct",   "qwen",    "~16 GB · 需要独显"),
    ("Qwen2.5-14B-Instruct",  "qwen",    "~32 GB · 高端 GPU"),
    ("Llama-3.2-1B-Instruct", "llama3",  "~3 GB · Mac M4 ✅"),
    ("Llama-3.2-3B-Instruct", "llama3",  "~8 GB · Mac M4 ✅"),
    ("SmolLM2-1.7B-Instruct", "default", "~4 GB · Mac M4 ✅"),
    ("Phi-3.5-mini-instruct", "phi",     "~8 GB · Mac M4 ✅"),
]

# 模板映射（模型名 → 模板名）
_MODEL_TEMPLATE_MAP = {preset[0]: preset[1] for preset in MODEL_PRESETS}


def get_template_for_model(model_name_or_path: str) -> str:
    """根据模型路径/名称自动推断对应的对话模板"""
    name = Path(model_name_or_path).name
    for model_key, template in _MODEL_TEMPLATE_MAP.items():
        if model_key.lower() in name.lower():
            return template
    # 按关键词兜底匹配
    name_lower = name.lower()
    if "qwen" in name_lower:
        return "qwen"
    if "llama" in name_lower:
        return "llama3"
    if "phi" in name_lower:
        return "phi"
    if "chatglm" in name_lower:
        return "chatglm3"
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

    def auto_adjust(self) -> "TrainConfig":
        """根据当前设备自动调整精度和 Flash Attention 设置"""
        info = get_device_info()
        self.fp16 = info["use_fp16"]
        self.bf16 = info["use_bf16"]
        self.flash_attn = info["flash_attn"]
        return self


# ──────────────────────────────────────────────────────────
#  LLaMA-Factory 接口
# ──────────────────────────────────────────────────────────

def check_llamafactory() -> tuple[bool, str]:
    """检测 LLaMA-Factory 是否已安装"""
    cli = shutil.which("llamafactory-cli")
    if cli:
        return True, cli

    result = subprocess.run(
        [sys.executable, "-m", "llamafactory.cli", "--help"],
        capture_output=True,
        timeout=10,
    )
    if result.returncode == 0:
        return True, f"{sys.executable} -m llamafactory.cli"

    return False, ""


def _build_dataset_info(config: TrainConfig) -> dict:
    return {
        config.dataset_name: {
            "file_name": Path(config.dataset_path).name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "response": "output",
                "system": "system",
            },
        }
    }


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
        "overwrite_output_dir": config.overwrite_output_dir,
        "default_system": config.default_system,
        "trust_remote_code": True,
        # 禁用 wandb / tensorboard 等实验追踪，避免需要登录 API key
        "report_to": "none",
    }
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
    "Qwen2.5-0.5B-Instruct":  ("Qwen/Qwen2.5-0.5B-Instruct",                  "Qwen/Qwen2.5-0.5B-Instruct"),
    "Qwen2.5-1.5B-Instruct":  ("Qwen/Qwen2.5-1.5B-Instruct",                  "Qwen/Qwen2.5-1.5B-Instruct"),
    "Qwen2.5-3B-Instruct":    ("Qwen/Qwen2.5-3B-Instruct",                    "Qwen/Qwen2.5-3B-Instruct"),
    "Qwen2.5-7B-Instruct":    ("Qwen/Qwen2.5-7B-Instruct",                    "Qwen/Qwen2.5-7B-Instruct"),
    "Qwen2.5-14B-Instruct":   ("Qwen/Qwen2.5-14B-Instruct",                   "Qwen/Qwen2.5-14B-Instruct"),
    "Llama-3.2-1B-Instruct":  ("LLM-Research/Meta-Llama-3.2-1B-Instruct",     "meta-llama/Llama-3.2-1B-Instruct"),
    "Llama-3.2-3B-Instruct":  ("LLM-Research/Meta-Llama-3.2-3B-Instruct",     "meta-llama/Llama-3.2-3B-Instruct"),
    "SmolLM2-1.7B-Instruct":  ("HuggingFaceTB/SmolLM2-1.7B-Instruct",        "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    "Phi-3.5-mini-instruct":  ("microsoft/Phi-3.5-mini-instruct",             "microsoft/Phi-3.5-mini-instruct"),
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

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
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
        ok, cli_path = check_llamafactory()
        if not ok:
            log_callback("❌ 未找到 LLaMA-Factory，请先运行：pip install llamafactory")
            return False

        # 自动适配设备参数
        config.auto_adjust()
        prepare_train_files(config)

        args = _build_train_args(config)
        cmd = (
            ["llamafactory-cli", "train"]
            if "llamafactory-cli" in cli_path
            else [sys.executable, "-m", "llamafactory.cli", "train"]
        )
        for k, v in args.items():
            cmd += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # 禁用 wandb 自动登录，避免没有 API key 时训练崩溃
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
        }
        self._stopped = False

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
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
