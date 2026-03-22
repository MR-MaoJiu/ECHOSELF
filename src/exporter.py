"""
模型导出模块
支持三个步骤：
  1. 合并 LoRA adapter → 完整 HuggingFace 模型（safetensors）
  2. 转换 GGUF 格式（供 Ollama / llama.cpp 使用）
  3. 导入 Ollama（生成 Modelfile 并调用 ollama create）
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from src.trainer import (
    _extend_env_for_windows_console_utf8,
    build_llamafactory_cli_argv,
    check_llamafactory,
)

# 与 trainer.PROJECT_ROOT 一致：合并/导出时相对路径以项目根为基准
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────
#  环境检测
# ──────────────────────────────────────────────────────────

def check_llamafactory_export() -> tuple[bool, str]:
    """检测当前 Python 中 LLaMA-Factory 是否可用（用于合并 LoRA）"""
    return check_llamafactory()


def check_llama_cpp() -> tuple[bool, str, str]:
    """
    检测 llama.cpp 的 GGUF 转换工具是否可用。
    返回 (可用, 方式, convert_hf_to_gguf.py 路径)
    """
    try:
        # 1. shutil.which 查找 convert_hf_to_gguf.py（brew 安装后在 /opt/homebrew/bin/）
        which_convert = shutil.which("convert_hf_to_gguf.py")
        if which_convert:
            return True, "brew", which_convert

        # 2. 常见固定路径直接检查
        for candidate in [
            "/opt/homebrew/bin/convert_hf_to_gguf.py",
            "/usr/local/bin/convert_hf_to_gguf.py",
            "/opt/homebrew/Cellar/llama.cpp/bin/convert_hf_to_gguf.py",
        ]:
            if Path(candidate).exists():
                return True, "brew", candidate

        # 3. find 命令查找（慢一些，兜底）
        try:
            result = subprocess.run(
                ["find", "/opt/homebrew", "-name", "convert_hf_to_gguf.py", "-maxdepth", "8"],
                capture_output=True, text=True, timeout=8,
            )
            found = [l for l in result.stdout.strip().splitlines() if l.strip()]
            if found:
                return True, "brew", found[0]
        except Exception:
            pass

        # 4. llama-cpp-python Python 包（不支持命令行转换，提示用 brew）
        if importlib.util.find_spec("llama_cpp") is not None:
            return True, "python", "llama-cpp-python"

        return False, "", ""
    except Exception:
        return False, "", ""


def check_ollama() -> tuple[bool, str]:
    """检测 ollama 是否已安装"""
    cli = shutil.which("ollama")
    if cli:
        return True, cli
    return False, ""


# ──────────────────────────────────────────────────────────
#  Step 1：合并 LoRA → 完整模型
# ──────────────────────────────────────────────────────────

class MergeProcess:
    """使用 LLaMA-Factory export 命令合并 LoRA adapter 到基础模型"""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stopped = False

    def start(
        self,
        base_model: str,
        adapter_path: str,
        template: str,
        output_dir: str,
        log_callback: Callable[[str], None],
        done_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        ok, _cli_path = check_llamafactory_export()
        if not ok:
            log_callback(
                f"❌ 当前 Python 中未安装 LLaMA-Factory，请执行："
                f"`{sys.executable} -m pip install llamafactory`"
            )
            return False

        if not Path(base_model).exists():
            log_callback(f"❌ 基础模型路径不存在：{base_model}")
            return False
        if not Path(adapter_path).exists():
            log_callback(f"❌ Adapter 路径不存在：{adapter_path}")
            return False

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        export_args = {
            "model_name_or_path": base_model,
            "adapter_name_or_path": adapter_path,
            "template": template,
            "finetuning_type": "lora",
            "export_dir": output_dir,
            "export_size": 2,
            "export_legacy_format": False,
        }
        cmd = build_llamafactory_cli_argv("export", export_args)

        self._stopped = False
        env = _extend_env_for_windows_console_utf8(
            {**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        # 与训练子进程一致：stdin 关闭、行缓冲；不在 Windows 上使用 CREATE_NO_WINDOW，避免无日志或异常退出
        popen_kw = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )
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


# ──────────────────────────────────────────────────────────
#  Step 2：转换 GGUF
# ──────────────────────────────────────────────────────────

# 量化精度选项及说明
QUANT_OPTIONS: list[tuple[str, str]] = [
    ("Q4_K_M", "4-bit 量化（推荐）— 体积最小，质量好，Ollama 默认"),
    ("Q5_K_M", "5-bit 量化 — 体积略大，精度稍高"),
    ("Q8_0",   "8-bit 量化 — 接近原始精度，体积约为 fp16 一半"),
    ("F16",    "半精度浮点 — 无损精度，体积较大"),
]


class GgufProcess:
    """将合并后的 HuggingFace 模型转换为 GGUF 格式"""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stopped = False

    def start(
        self,
        merged_dir: str,
        output_path: str,
        quant_type: str,
        log_callback: Callable[[str], None],
        done_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        ok, method, path = check_llama_cpp()
        if not ok:
            log_callback(
                "❌ 未找到 GGUF 转换工具。\n\n"
                "请选择一种方式安装：\n"
                "  方式1（推荐 Mac）：brew install llama.cpp\n"
                "  方式2（Python）：pip install llama-cpp-python\n\n"
                "安装后重启应用，重新点击转换。"
            )
            return False

        if not Path(merged_dir).exists():
            log_callback(f"❌ 合并模型目录不存在：{merged_dir}")
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._stopped = False
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        if method == "python":
            log_callback("⚠️ 检测到 llama-cpp-python，但转换需要 llama.cpp CLI 工具")
            log_callback("请运行：brew install llama.cpp")
            return False

        # 使用 convert_hf_to_gguf.py 脚本
        # 先生成 f16 gguf，再量化
        tmp_f16 = str(Path(output_path).parent / "tmp_f16.gguf")
        cmd_convert = [sys.executable, path, merged_dir, "--outfile", tmp_f16, "--outtype", "f16"]

        log_callback(f"▶ 开始转换为 F16 GGUF：{tmp_f16}")
        log_callback(f"  命令：{' '.join(cmd_convert)}\n")

        try:
            self._process = subprocess.Popen(
                cmd_convert,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
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
            code = self._process.returncode

            if code != 0 or self._stopped:
                if done_callback:
                    done_callback(code)
                return

            # 如果目标格式不是 f16，继续量化
            if quant_type.upper() == "F16":
                shutil.move(tmp_f16, output_path)
                log_callback(f"\n✅ 已保存 F16 GGUF：{output_path}")
                if done_callback:
                    done_callback(0)
                return

            # 量化步骤
            llama_quantize = shutil.which("llama-quantize")
            if not llama_quantize:
                log_callback("⚠️ 未找到 llama-quantize，跳过量化，使用 F16 格式")
                shutil.move(tmp_f16, output_path)
                if done_callback:
                    done_callback(0)
                return

            log_callback(f"\n▶ 开始量化为 {quant_type}：{output_path}")
            quant_proc = subprocess.run(
                [llama_quantize, tmp_f16, output_path, quant_type.upper()],
                capture_output=False,
                text=True,
            )
            try:
                Path(tmp_f16).unlink(missing_ok=True)
            except Exception:
                pass
            if quant_proc.returncode == 0:
                log_callback(f"\n✅ 量化完成：{output_path}")
            else:
                log_callback(f"❌ 量化失败（code={quant_proc.returncode}）")
            if done_callback:
                done_callback(quant_proc.returncode)

        threading.Thread(target=_stream, daemon=True).start()
        return True

    def stop(self):
        self._stopped = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None


# ──────────────────────────────────────────────────────────
#  Step 3：导入 Ollama
# ──────────────────────────────────────────────────────────

def generate_modelfile(
    gguf_path: str,
    model_name: str,
    system_prompt: str,
    temperature: float = 0.7,
) -> str:
    """生成 Ollama Modelfile 内容"""
    abs_path = str(Path(gguf_path).resolve())
    return (
        f"FROM {abs_path}\n\n"
        f'SYSTEM """\n{system_prompt}\n"""\n\n'
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER top_p 0.9\n"
        f"PARAMETER repeat_penalty 1.1\n"
    )


def save_modelfile(modelfile_content: str, save_path: str) -> str:
    """将 Modelfile 内容写入文件，返回文件路径"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    return save_path


class OllamaImportProcess:
    """调用 ollama create 将 GGUF 导入 Ollama"""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stopped = False

    def start(
        self,
        model_name: str,
        modelfile_path: str,
        log_callback: Callable[[str], None],
        done_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        ok, cli = check_ollama()
        if not ok:
            log_callback(
                "❌ 未找到 ollama。\n\n"
                "请先安装 Ollama：https://ollama.com/download\n"
                "安装后重新点击导入。"
            )
            return False

        # 转为绝对路径，避免子进程 cwd 不同导致找不到文件
        modelfile_abs = str(Path(modelfile_path).resolve())
        if not Path(modelfile_abs).exists():
            log_callback(f"❌ Modelfile 不存在：{modelfile_abs}，请先点击「生成 Modelfile」")
            return False

        # Ollama 模型名只允许小写字母、数字、连字符、冒号，自动转换
        import re as _re
        safe_name = _re.sub(r"[^a-z0-9\-:]", "-", model_name.lower()).strip("-")
        if not safe_name:
            safe_name = "echoself"
        if safe_name != model_name:
            log_callback(f"⚠️ 模型名已自动转换：{model_name} → {safe_name}（Ollama 只支持小写字母/数字/连字符）\n")

        cmd = [cli, "create", safe_name, "-f", modelfile_abs]
        log_callback(f"▶ 执行：{' '.join(cmd)}\n")

        self._stopped = False
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
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
            code = self._process.returncode
            if code == 0:
                log_callback(
                    f"\n✅ 导入成功！\n\n"
                    f"现在可以在终端运行：\n"
                    f"  ollama run {model_name}\n\n"
                    f"或在任何支持 Ollama 的客户端（如 Open WebUI、ChatBox）中选择 {model_name}"
                )
            else:
                log_callback(f"❌ 导入失败（code={code}）")
            if done_callback:
                done_callback(code)

        threading.Thread(target=_stream, daemon=True).start()
        return True

    def stop(self):
        self._stopped = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
