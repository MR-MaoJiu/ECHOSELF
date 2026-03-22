"""
EchoSelf — 从聊天记录训练数字分身
Gradio GUI 主程序
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import (
    chat_stream,
    get_loaded_info,
    is_loaded,
    load_model,
    scan_local_adapters,
    unload_model,
)
from src.exporter import (
    GgufProcess,
    MergeProcess,
    OllamaImportProcess,
    QUANT_OPTIONS,
    check_llama_cpp,
    check_ollama,
    generate_modelfile,
    save_modelfile,
)
from src.parser import get_message_type_stats, parse_multiple_files
from src.preprocessor import build_qa_pairs, save_dataset
from src.trainer import (
    MODEL_DOWNLOAD_IDS,
    MODEL_PRESETS,
    _abs_project_path,
    get_model_preset_choices,
    DownloadProcess,
    TrainConfig,
    TrainingProcess,
    check_huggingface_hub,
    check_llamafactory,
    check_modelscope,
    get_device_info,
    get_template_for_model,
    get_train_command,
    has_training_artifacts,
    scan_checkpoints,
)

OUTPUT_DIR = Path("./output")
DATASET_PATH = OUTPUT_DIR / "sft_data.json"
# 实时训练日志持久化文件（刷新后可恢复）
TRAIN_LOG_FILE = OUTPUT_DIR / "train_log_live.txt"

_training_process  = TrainingProcess()
_merge_process     = MergeProcess()
_gguf_process      = GgufProcess()
_ollama_process    = OllamaImportProcess()
_train_logs: list[str] = []
_metrics_history: list[dict] = []   # 记录每步 {step, loss} 用于绘图

_download_process = DownloadProcess()
_download_logs: list[str] = []

# 启动时检测一次设备及下载工具
_DEVICE_INFO = get_device_info()
_MS_OK  = check_modelscope()
_HF_OK  = check_huggingface_hub()


# ──────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────

def _open_folder_picker_tkinter_subprocess() -> str:
    """
    Windows / Linux：在独立子进程中用 tkinter 选文件夹（与 Gradio 线程隔离）。
    用户取消返回空字符串。
    """
    import subprocess
    import sys

    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk()\n"
        "r.withdraw()\n"
        "r.attributes('-topmost', True)\n"
        "try:\n"
        "    p = filedialog.askdirectory(title='选择聊天记录文件夹')\n"
        "finally:\n"
        "    r.destroy()\n"
        "print(p or '', end='')"
    )
    try:
        run_kw = {
            "args": [sys.executable, "-c", code],
            "capture_output": True,
            "text": True,
            "timeout": 300,
        }
        if sys.platform == "win32":
            run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(**run_kw)
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except Exception:
        return ""


def _open_folder_picker() -> str:
    """
    跨平台弹出文件夹选择对话框。
    macOS：osascript 原生对话框；Windows / Linux：tkinter（子进程）。
    用户取消返回空字符串。
    """
    import subprocess
    import sys

    if sys.platform == "darwin":
        script = 'POSIX path of (choose folder with prompt "选择聊天记录文件夹")'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return result.stdout.strip().rstrip("/")
            return ""
        except Exception:
            return ""

    return _open_folder_picker_tkinter_subprocess()


def _detect_my_id_from_folder(folder: Path) -> str:
    """扫描文件夹内第一个 JSON 文件，从 isSend=1 的消息中推断账号 ID"""
    import json as _json
    for fp in sorted(folder.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = _json.load(f)
            for msg in data.get("messages", []):
                if msg.get("isSend") == 1:
                    uid = msg.get("senderUsername", "").strip()
                    if uid:
                        return uid
        except Exception:
            continue
    return ""


def pick_folder(current_my_id: str) -> tuple[str, str, str]:
    """
    弹出文件夹选择器，选定后自动检测账号 ID。
    返回 (folder_path, my_id, hint_markdown)
    """
    folder_str = _open_folder_picker()
    if not folder_str:
        return "", current_my_id, ""

    folder = Path(folder_str)
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        return folder_str, current_my_id, f"⚠️ 文件夹中未找到 `.json` 文件：`{folder_str}`"

    # 仅在用户未手动填写时才自动检测
    detected_id = current_my_id.strip()
    if not detected_id:
        detected_id = _detect_my_id_from_folder(folder)

    id_hint = f"，已识别账号 ID：`{detected_id}`" if detected_id else "，**未找到已发消息**，可手动填写账号 ID"
    hint = f"✅ 已选择：`{folder_str}`\n\n共找到 **{len(json_files)}** 个 JSON 文件{id_hint}"
    return folder_str, detected_id, hint


def _collect_json_files(folder_path: str) -> tuple[list[Path], str]:
    folder = Path(folder_path.strip())
    if not folder_path.strip():
        return [], "⚠️ 请输入聊天记录文件夹路径"
    if not folder.exists():
        return [], f"❌ 路径不存在：`{folder}`"
    if not folder.is_dir():
        return [], f"❌ 该路径不是文件夹：`{folder}`"
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        return [], f"❌ 文件夹中没有 `.json` 文件：`{folder}`"
    return json_files, ""


def _fmt_stats(result, contact_name: str) -> str:
    return "\n".join([
        f"### {contact_name}",
        "",
        "| 指标 | 数量 |",
        "|------|------|",
        f"| 原始消息总数 | {result.total_messages} |",
        f"| 有效保留消息 | {result.kept_messages} |",
        f"| 跳过（类型过滤） | {result.skipped_type} |",
        f"| 跳过（长度过滤） | {result.skipped_length} |",
        f"| PII 脱敏条数 | {result.pii_removed} |",
        f"| 禁用词过滤条数 | {result.blocked_removed} |",
        f"| **生成 QA 对数** | **{len(result.qa_pairs)}** |",
    ])


def _preview_qa(qa_pairs: list, n: int = 5) -> str:
    if not qa_pairs:
        return "⚠️ 未生成任何 QA 对，请检查数据和参数配置。"
    lines = [f"### 数据预览（前 {min(n, len(qa_pairs))} 条）\n"]
    for i, pair in enumerate(qa_pairs[:n]):
        q = pair.instruction[:120] + ("..." if len(pair.instruction) > 120 else "")
        a = pair.output[:120] + ("..." if len(pair.output) > 120 else "")
        lines += [f"**[{i+1}] {pair.time}**", f"> 对方：{q}", f"> 我：{a}", ""]
    return "\n".join(lines)


def scan_local_models() -> list[str]:
    """
    扫描 ./models 目录，返回包含 config.json 的子目录路径列表。
    config.json 是 HuggingFace / ModelScope 模型的标志文件。
    """
    models_dir = Path("./models")
    if not models_dir.exists():
        return []
    return [
        str(d)
        for d in sorted(models_dir.iterdir())
        if d.is_dir() and (d / "config.json").exists()
    ]


def _device_banner() -> str:
    info = _DEVICE_INFO
    lines = [
        f"{info['icon']} **当前设备：{info['device']}**",
        "",
        f"📌 {info['notes']}",
        "",
        "**推荐模型：** " + "、".join(info["recommended_models"]),
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
#  数据处理回调
# ──────────────────────────────────────────────────────────

def preview_type_stats(folder_path: str, my_id: str) -> tuple[str, str]:
    """
    扫描文件夹消息类型分布，同时返回检测到的账号 ID。
    Returns: (stats_markdown, detected_my_id)
    """
    json_files, err = _collect_json_files(folder_path)
    if err:
        return err, my_id
    sessions = parse_multiple_files(json_files, my_id.strip())
    if not sessions:
        return "❌ 解析失败，请确认文件格式正确", my_id

    # 从解析结果中取出实际使用的 my_id（parser 已做自动推断）
    detected_id = sessions[0].my_id if sessions and sessions[0].my_id else my_id

    lines = []
    for s in sessions:
        stats = get_message_type_stats(s)
        lines.append(f"**{s.contact_name}** ({s.chat_type})  共 {len(s.messages)} 条消息")
        for t, c in list(stats.items())[:10]:
            lines.append(f"  - {t}：{c} 条")
        lines.append("")

    if detected_id and not my_id.strip():
        lines.append(f"---\n✅ 已自动识别账号 ID：`{detected_id}`")

    return "\n".join(lines), detected_id


def process_data(
    folder_path, my_id, time_window, combine_window,
    min_len, max_len, blocked_words_text, enable_pii,
    system_prompt, output_format,
) -> tuple[str, str, str | None]:
    json_files, err = _collect_json_files(folder_path)
    if err:
        return err, "", None

    blocked_words = [w.strip() for w in blocked_words_text.strip().splitlines() if w.strip()]
    sessions = parse_multiple_files(json_files, my_id.strip())
    if not sessions:
        return "❌ 解析失败，请检查文件格式", "", None

    all_qa_pairs = []
    stats_blocks = []

    for session in sessions:
        result = build_qa_pairs(
            session=session,
            time_window_minutes=int(time_window),
            single_combine_window_minutes=int(combine_window),
            min_msg_len=int(min_len),
            max_msg_len=int(max_len),
            blocked_words=blocked_words,
            enable_pii_removal=enable_pii,
            system_prompt=system_prompt,
        )
        all_qa_pairs.extend(result.qa_pairs)
        stats_blocks.append(_fmt_stats(result, session.contact_name))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = save_dataset(all_qa_pairs, DATASET_PATH, fmt=output_format)

    stats_text = "\n\n---\n\n".join(stats_blocks)
    stats_text += (
        f"\n\n---\n\n✅ **共处理 {len(json_files)} 个文件，"
        f"保存 {saved_count} 条训练数据** → `{DATASET_PATH}`"
    )
    return stats_text, _preview_qa(all_qa_pairs), str(DATASET_PATH)


# ──────────────────────────────────────────────────────────
#  训练回调
# ──────────────────────────────────────────────────────────

import re as _re


def _parse_train_metrics(line: str, metrics: dict) -> None:
    """
    从单行日志中提取训练指标，就地更新 metrics 字典。
    支持 transformers Trainer 的 JSON dict 输出和 tqdm 进度条。
    """
    # transformers 格式: {'loss': 0.56, 'learning_rate': 1e-4, 'epoch': 0.5}
    for key, pattern in [
        ("loss",  r"['\"]loss['\"]\s*:\s*([\d.]+)"),
        ("lr",    r"['\"]learning_rate['\"]\s*:\s*([\d.e+\-]+)"),
        ("epoch", r"['\"]epoch['\"]\s*:\s*([\d.]+)"),
        ("grad",  r"['\"]grad_norm['\"]\s*:\s*([\d.]+)"),
    ]:
        m = _re.search(pattern, line)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass

    # tqdm 进度条: 50%|████| 5/10 [00:10<00:10, 2.00it/s]
    tqdm_m = _re.search(
        r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:?]+),\s*([\d.]+)\s*it/s",
        line,
    )
    if tqdm_m:
        metrics["pct"]         = int(tqdm_m.group(1))
        metrics["step"]        = int(tqdm_m.group(2))
        metrics["total_steps"] = int(tqdm_m.group(3))
        metrics["elapsed"]     = tqdm_m.group(4)
        metrics["remain"]      = tqdm_m.group(5)
        metrics["speed"]       = float(tqdm_m.group(6))

    # 训练结束汇总行: train_loss = 0.4532
    final_m = _re.search(r"train_loss\s*=\s*([\d.]+)", line)
    if final_m:
        metrics["final_loss"] = float(final_m.group(1))


def _make_loss_plot(history: list[dict]):
    """根据历史指标生成 matplotlib Loss 曲线图，history 为 [{step, loss}, ...]"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        if len(history) < 2:
            return None

        steps  = np.array([h["step"] for h in history], dtype=float)
        losses = np.array([h["loss"] for h in history], dtype=float)

        # ── 颜色主题 ────────────────────────────────────────────
        BG       = "#0d1117"
        PANEL    = "#161b22"
        GRID     = "#21262d"
        TEXT     = "#c9d1d9"
        SUBTEXT  = "#6e7681"
        ACCENT   = "#58a6ff"   # 蓝
        ACCENT2  = "#3fb950"   # 绿（最低点）
        DANGER   = "#f85149"   # 红（最高点）

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        # ── 原始数据散点（淡显，不连线） ────────────────────────
        ax.scatter(steps, losses, s=3, color=ACCENT, alpha=0.18, zorder=1, linewidths=0)

        # ── 滑动平均平滑线 ──────────────────────────────────────
        window = min(max(len(losses) // 8, 5), 30)
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        sx = steps[window - 1:]

        # 渐变填充：从曲线到底部
        ax.fill_between(sx, smooth, smooth.min() - 0.1,
                        alpha=0.15, color=ACCENT, zorder=2)

        # 主曲线
        ax.plot(sx, smooth, color=ACCENT, linewidth=2.2, zorder=3, solid_capstyle="round")

        # ── 标注最低点 ──────────────────────────────────────────
        min_idx = np.argmin(smooth)
        ax.scatter([sx[min_idx]], [smooth[min_idx]], s=60, color=ACCENT2,
                   zorder=5, linewidths=0)
        ax.annotate(
            f"最低 {smooth[min_idx]:.3f}",
            xy=(sx[min_idx], smooth[min_idx]),
            xytext=(8, 8), textcoords="offset points",
            color=ACCENT2, fontsize=8.5, fontweight="bold",
        )

        # ── 标注最新值（右端） ──────────────────────────────────
        ax.annotate(
            f"当前 {smooth[-1]:.3f}",
            xy=(sx[-1], smooth[-1]),
            xytext=(-6, 10), textcoords="offset points",
            color=TEXT, fontsize=8.5,
            ha="right",
        )

        # ── 轴与网格 ────────────────────────────────────────────
        ax.set_xlabel("训练步数 (Step)", color=SUBTEXT, fontsize=9.5, labelpad=6)
        ax.set_ylabel("Loss", color=SUBTEXT, fontsize=9.5, labelpad=6)
        ax.set_title("Loss 训练曲线", color=TEXT, fontsize=12,
                     fontweight="bold", pad=10)
        ax.tick_params(colors=SUBTEXT, labelsize=8.5, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, alpha=0.35, color=GRID, linestyle="-", linewidth=0.8)
        ax.set_xlim(steps[0], steps[-1])
        y_lo = max(0, losses.min() - 0.3)
        y_hi = losses.max() + 0.3
        ax.set_ylim(y_lo, y_hi)

        # ── 右侧进度文字 ────────────────────────────────────────
        total = history[-1].get("total_steps") or int(steps[-1])
        pct   = round(steps[-1] / total * 100, 1) if total else 0
        fig.text(0.99, 0.97, f"{int(steps[-1])} / {total} steps  {pct}%",
                 ha="right", va="top", color=SUBTEXT, fontsize=8.5,
                 transform=fig.transFigure)

        fig.tight_layout(pad=0.8)
        plt.close(fig)
        return fig
    except Exception:
        return None


def _loss_trend(history: list[dict]) -> str:
    """根据最近 10 步 loss 判断训练健康度"""
    recent = [h["loss"] for h in history[-10:] if "loss" in h]
    if len(recent) < 4:
        return ""
    delta = recent[-1] - recent[0]
    if delta < -0.05:
        return "**状态** 🟢 Loss 持续下降，训练正常"
    elif delta > 0.05:
        return "**状态** 🔴 Loss 上升，建议降低学习率或停止检查"
    elif abs(delta) < 0.01:
        return "**状态** 🟡 Loss 趋于平稳，可能接近收敛或陷入平台"
    return "**状态** 🔵 训练中..."


def _format_progress(metrics: dict, history: list = None, done: bool = False, status: str = "") -> str:
    """将 metrics 字典格式化为 Markdown 进度面板，可选传入 history 进行趋势判断"""
    if not metrics and not done:
        return "_⏳ 等待训练输出..._"

    rows = []

    # 文字进度条
    pct = metrics.get("pct", 0)
    if pct or "step" in metrics:
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        rows.append(f"`{bar}` **{pct}%**")

    if "step" in metrics:
        rows.append(f"**步数** &nbsp; {metrics['step']} / {metrics.get('total_steps', '?')}")
    if "epoch" in metrics:
        rows.append(f"**Epoch** &nbsp; {metrics['epoch']:.2f}")
    if "loss" in metrics:
        rows.append(f"**Loss** &nbsp; `{metrics['loss']:.4f}`")
    if "grad" in metrics:
        rows.append(f"**Grad Norm** &nbsp; `{metrics['grad']:.4f}`")
    if "lr" in metrics:
        rows.append(f"**学习率** &nbsp; `{metrics['lr']:.2e}`")
    if "speed" in metrics:
        rows.append(f"**速度** &nbsp; {metrics['speed']:.1f} it/s")
    if "elapsed" in metrics:
        rows.append(f"**用时** &nbsp; {metrics['elapsed']} &nbsp;·&nbsp; **剩余** {metrics.get('remain', '?')}")
    if "final_loss" in metrics:
        rows.append(f"**最终 Loss** &nbsp; `{metrics['final_loss']:.4f}`")

    # 训练健康度（有历史数据时才显示）
    if history:
        trend = _loss_trend(history)
        if trend:
            rows.append(trend)

    if status:
        rows.append(f"\n{status}")

    return "  \n".join(rows) if rows else "_⏳ 等待训练输出..._"


def on_model_change(model_name: str) -> str:
    """选择模型后自动填写对话模板"""
    return get_template_for_model(model_name)


def get_command_preview(
    model_name, model_path, template, dataset_path, output_dir,
    system_prompt, lora_rank, epochs, batch_size, grad_accum,
    learning_rate, cutoff_len,
) -> str:
    cfg = _make_config(
        model_name, model_path, template, dataset_path, output_dir,
        system_prompt, lora_rank, epochs, batch_size, grad_accum,
        learning_rate, cutoff_len,
    )
    cfg.auto_adjust()
    return f"```bash\n{get_train_command(cfg)}\n```"


def _make_config(
    model_name, model_path, template, dataset_path, output_dir,
    system_prompt, lora_rank, epochs, batch_size, grad_accum,
    learning_rate, cutoff_len,
    resume_checkpoint: str = "",
) -> TrainConfig:
    # 自定义路径优先，否则直接用 model_name（现在已是本地路径）
    resolved_path = model_path.strip() if model_path.strip() else model_name
    return TrainConfig(
        model_name_or_path=resolved_path,
        template=template,
        dataset_path=dataset_path,
        dataset_dir=str(Path(dataset_path).parent),
        output_dir=output_dir,
        default_system=system_prompt,
        lora_rank=int(lora_rank),
        num_train_epochs=float(epochs),
        per_device_train_batch_size=int(batch_size),
        gradient_accumulation_steps=int(grad_accum),
        learning_rate=float(learning_rate),
        cutoff_len=int(cutoff_len),
        resume_from_checkpoint=(resume_checkpoint or "").strip() or None,
    )


def start_training(
    model_name, model_path, template, dataset_path, output_dir,
    system_prompt, lora_rank, epochs, batch_size, grad_accum,
    learning_rate, cutoff_len,
    resume_checkbox: bool = False,
    resume_checkpoint: str = "",
):
    """
    训练生成器：每 0.5 秒 yield 一次 (log_text, progress_md)。
    log_text 为原始日志，progress_md 为解析后的结构化进度面板。
    """
    global _train_logs, _metrics_history
    _train_logs = []
    _metrics_history = []
    _metrics: dict = {}
    _last_plot_step: list = [0]   # 用列表绕过闭包只读限制

    # 防止训练进行中重复点击开始
    if _training_process.is_running:
        yield (
            "⚠️ 训练进程仍在运行中，请先点击「⏹️ 停止训练」后再重新开始。\n\n"
            "若需查看当前进度，请直接观察下方日志，或刷新页面后点「📋 恢复上次训练记录」。",
            "",
            None,
        )
        return

    if not Path(_abs_project_path(dataset_path)).exists():
        yield "❌ 训练数据文件不存在，请先完成数据处理步骤。", "", None
        return

    # 续训时使用勾选的 checkpoint，否则全新训练（resume_checkpoint 可能为 None）
    _ckpt_str = (resume_checkpoint or "").strip()
    ckpt = _ckpt_str if (resume_checkbox and _ckpt_str) else ""
    cfg = _make_config(
        model_name, model_path, template, dataset_path, output_dir,
        system_prompt, lora_rank, epochs, batch_size, grad_accum,
        learning_rate, cutoff_len,
        resume_checkpoint=ckpt,
    )

    # 训练开始时清空并初始化日志持久化文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(TRAIN_LOG_FILE, "w", encoding="utf-8", buffering=1)

    done_flag: dict = {"done": False, "code": -1}

    def on_log(line: str):
        _train_logs.append(line)
        # 实时写入持久化文件，浏览器刷新后可恢复
        _log_file.write(line + "\n")
        _log_file.flush()
        # stdout 日志只解析 epoch/lr/grad 等辅助指标，step+loss 由 trainer_log.jsonl 提供
        _parse_train_metrics(line, _metrics)

    def on_done(code: int):
        done_flag["done"] = True
        done_flag["code"] = code

    started = _training_process.start(cfg, on_log, on_done)
    if not started:
        yield "\n".join(_train_logs), "", None
        return

    import time
    import json as _json

    # LLaMA-Factory 把 step/loss 写入 trainer_log.jsonl，从这里读取最准确（与 trainer 一致用项目根解析相对路径）
    _jsonl_path = Path(_abs_project_path(output_dir)) / "trainer_log.jsonl"
    _last_jsonl_size = [0]   # 记录上次读取的文件大小，避免重复解析

    def _sync_metrics_from_jsonl():
        """增量读取 trainer_log.jsonl，同步 _metrics_history 和 _metrics"""
        if not _jsonl_path.exists():
            return
        try:
            size = _jsonl_path.stat().st_size
            if size <= _last_jsonl_size[0]:
                return
            with open(_jsonl_path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            _last_jsonl_size[0] = size
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                step = rec.get("current_steps")
                loss = rec.get("loss")
                if step and loss:
                    # 去重插入
                    if not _metrics_history or _metrics_history[-1].get("step") != step:
                        _metrics_history.append({"step": step, "loss": loss})
                    # 同步到 _metrics 供进度面板使用
                    _metrics["step"]        = step
                    _metrics["loss"]        = loss
                    _metrics["total_steps"] = rec.get("total_steps", 0)
                    _metrics["pct"]         = rec.get("percentage", 0)
                    _metrics["elapsed"]     = rec.get("elapsed_time", "")
                    _metrics["remain"]      = rec.get("remaining_time", "")
                    if "lr" in rec:
                        _metrics["lr"] = rec["lr"]
                    if "epoch" in rec:
                        _metrics["epoch"] = rec["epoch"]
        except Exception:
            pass

    _plot_cache = [None]   # 缓存上一次图，减少重复绘制
    while not done_flag["done"]:
        time.sleep(0.5)
        # 从 trainer_log.jsonl 同步最新指标（这是 loss 数据的权威来源）
        _sync_metrics_from_jsonl()
        # 每新增 5 个数据点才重绘曲线（避免过于频繁）
        cur_step = len(_metrics_history)
        if cur_step >= _last_plot_step[0] + 5 or (cur_step > 0 and _plot_cache[0] is None):
            _plot_cache[0] = _make_loss_plot(_metrics_history)
            _last_plot_step[0] = cur_step
        yield (
            "\n".join(_train_logs[-150:]),
            _format_progress(_metrics, _metrics_history),
            _plot_cache[0],
        )

    rc = done_flag["code"]
    out_abs = _abs_project_path(output_dir)
    if rc == 0 and has_training_artifacts(output_dir):
        status = f"✅ 训练完成！模型输出目录：{out_abs}"
    elif rc == 0:
        status = (
            f"⚠️ 训练进程已结束（退出码 0），但未在输出目录发现 LoRA 产物（adapter_config.json 或 checkpoint-*）。\n"
            f"请查看上方日志，并确认目录：{out_abs}"
        )
    else:
        status = f"❌ 训练异常退出（code={rc}）"
    _train_logs.append(f"\n{status}")
    # 写入最终状态并关闭日志文件
    _log_file.write(f"\n{status}\n")
    _log_file.close()
    final_plot = _make_loss_plot(_metrics_history)
    yield (
        "\n".join(_train_logs[-150:]),
        _format_progress(_metrics, _metrics_history, done=True, status=status),
        final_plot,
    )


def stop_training() -> str:
    _training_process.stop()
    return "⏹ 已发送停止信号"


# ──────────────────────────────────────────────────────────
#  下载回调
# ──────────────────────────────────────────────────────────

def on_dl_model_change(choice: str, source: str) -> tuple[str, str]:
    """根据模型选择和下载源自动填写模型 ID 与本地路径"""
    model_name = choice.split("  ")[0]
    ids = MODEL_DOWNLOAD_IDS.get(model_name)
    if not ids:
        model_id = model_name
    elif "ModelScope" in source:
        model_id = ids[0]
    else:
        model_id = ids[1]
    local_dir = f"./models/{model_name}"
    return model_id, local_dir


def start_download(source: str, model_id: str, local_dir: str):
    global _download_logs
    _download_logs = []

    if not model_id.strip():
        yield "❌ 模型 ID 不能为空"
        return

    if not local_dir.strip():
        yield "❌ 本地路径不能为空"
        return

    # 检测依赖
    src_key = "modelscope" if "ModelScope" in source else "huggingface"
    if src_key == "modelscope" and not _MS_OK:
        yield "❌ 未安装 modelscope，请先运行：pip install modelscope"
        return
    if src_key == "huggingface" and not _HF_OK:
        yield "❌ 未安装 huggingface-hub，请先运行：pip install huggingface-hub"
        return

    _download_logs.append(f"⬇️  开始下载：{model_id}")
    _download_logs.append(f"📁  保存到：{local_dir}")
    _download_logs.append(f"🌐  下载源：{source}\n")

    done_flag: dict = {"done": False, "code": -1}

    def on_log(line: str):
        _download_logs.append(line)

    def on_done(code: int):
        done_flag["done"] = True
        done_flag["code"] = code

    started = _download_process.start(src_key, model_id.strip(), local_dir.strip(), on_log, on_done)
    if not started:
        yield "\n".join(_download_logs)
        return

    import time
    while not done_flag["done"]:
        time.sleep(1)
        yield "\n".join(_download_logs[-120:])

    suffix = "✅ 下载完成！" if done_flag["code"] == 0 else f"❌ 下载异常退出（code={done_flag['code']}）"
    _download_logs.append(f"\n{suffix}")
    yield "\n".join(_download_logs[-120:])


def stop_download() -> str:
    _download_process.stop()
    return "⏹ 已发送停止信号"


# ──────────────────────────────────────────────────────────
#  UI 构建
# ──────────────────────────────────────────────────────────

CSS = """
.tab-nav button { font-size: 15px; font-weight: 600; }

/* 设备信息面板 - 深色主题，高对比度 */
.device-info {
  background: #1c1c1e !important;
  border-radius: 12px !important;
  padding: 16px 20px !important;
  border: 1px solid #48484a !important;
  border-left: 4px solid #5e5ce6 !important;
}
.device-info p,
.device-info li,
.device-info span {
  color: #e5e5ea !important;
}
.device-info strong,
.device-info b {
  color: #a9a9fc !important;
}
.device-info code {
  background: #2c2c2e !important;
  color: #64d2ff !important;
  padding: 1px 5px;
  border-radius: 4px;
}

/* 下载日志框 */
.download-log textarea { font-family: monospace; font-size: 12px; }

/* 模型对话：加载状态面板 */
.inference-status {
  background: #0d1117 !important;
  border-radius: 10px !important;
  padding: 12px 18px !important;
  border: 1px solid #30363d !important;
  border-left: 4px solid #1f6feb !important;
  font-size: 13px !important;
}
.inference-status p, .inference-status span { color: #e6edf3 !important; }
.inference-status code { background: #161b22 !important; color: #79c0ff !important; border-radius: 4px; padding: 1px 5px; }

/* 帮助文档内容区 */
.help-content { max-width: 860px; margin: 0 auto; line-height: 1.9; }
.help-content h3 { color: #58a6ff !important; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
.help-content h4 { color: #79c0ff !important; }
.help-content code { background: #161b22 !important; color: #79c0ff !important; border-radius: 4px; padding: 1px 6px; }
.help-content blockquote { border-left: 3px solid #388bfd; padding-left: 12px; color: #8b949e !important; }

/* 训练实时进度面板 */
.train-progress {
  background: #0d1117 !important;
  border-radius: 10px !important;
  padding: 14px 20px !important;
  border: 1px solid #30363d !important;
  border-left: 4px solid #3fb950 !important;
  font-size: 14px !important;
  line-height: 1.8 !important;
}
.train-progress p,
.train-progress li,
.train-progress span {
  color: #e6edf3 !important;
}
.train-progress strong {
  color: #58a6ff !important;
}
.train-progress code {
  background: #161b22 !important;
  color: #79c0ff !important;
  padding: 1px 6px;
  border-radius: 4px;
}
.train-progress em {
  color: #8b949e !important;
}

/* 检测到 NVIDIA 显卡但 PyTorch 未启用 CUDA 时的顶部提示（高对比、易读） */
.cuda-setup-banner {
  border: 2px solid #e67e22 !important;
  border-radius: 12px !important;
  padding: 18px 22px !important;
  margin-bottom: 14px !important;
  background: linear-gradient(165deg, #2d333b 0%, #1c2128 55%, #161b22 100%) !important;
  border-left: 6px solid #ff922b !important;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
  font-size: 16px !important;
  line-height: 1.85 !important;
  color: #f6f8fa !important;
}
.cuda-setup-banner > div,
.cuda-setup-banner .prose,
.cuda-setup-banner .md,
.cuda-setup-banner article {
  color: #f6f8fa !important;
  font-size: inherit !important;
  line-height: inherit !important;
}
.cuda-setup-banner h1,
.cuda-setup-banner h2,
.cuda-setup-banner h3 {
  margin-top: 0.4em !important;
  margin-bottom: 0.65em !important;
  color: #ffd4a8 !important;
  font-weight: 700 !important;
  font-size: 1.28rem !important;
  letter-spacing: 0.02em !important;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.45) !important;
}
.cuda-setup-banner h3:first-of-type {
  margin-top: 0 !important;
}
.cuda-setup-banner p,
.cuda-setup-banner li,
.cuda-setup-banner ol,
.cuda-setup-banner ul {
  color: #f0f6fc !important;
  font-size: 1rem !important;
}
.cuda-setup-banner strong,
.cuda-setup-banner b {
  color: #ffffff !important;
  font-weight: 650 !important;
}
.cuda-setup-banner a {
  color: #7dd3fc !important;
  font-weight: 600 !important;
  text-decoration: underline !important;
  text-underline-offset: 3px !important;
}
.cuda-setup-banner a:hover {
  color: #bae6fd !important;
}
.cuda-setup-banner code {
  background: #0d1117 !important;
  color: #a5d6ff !important;
  padding: 2px 8px !important;
  border-radius: 6px !important;
  font-size: 0.9em !important;
  border: 1px solid #484f58 !important;
}
.cuda-setup-banner pre,
.cuda-setup-banner pre code {
  background: #0d1117 !important;
  color: #c9d1d9 !important;
  border: 1px solid #30363d !important;
}
.cuda-setup-banner blockquote {
  border-left: 4px solid #ff922b !important;
  margin: 0.75em 0 !important;
  padding: 10px 14px !important;
  background: rgba(13, 17, 23, 0.75) !important;
  border-radius: 0 8px 8px 0 !important;
}
.cuda-setup-banner blockquote,
.cuda-setup-banner blockquote p {
  color: #e6edf3 !important;
}

footer { display: none !important; }
"""

# 模型选项：展示文案随当前设备（显存/统一内存）动态标注「本机推荐」等
_MODEL_CHOICES = get_model_preset_choices(_DEVICE_INFO)
_MODEL_NAMES   = [name for name, _, _ in MODEL_PRESETS]
_DEFAULT_MODEL = _DEVICE_INFO["default_model"]
_DEFAULT_CHOICE = next(
    (c for c in _MODEL_CHOICES if _DEFAULT_MODEL in c),
    _MODEL_CHOICES[1],  # fallback: 1.5B
)


def build_ui() -> gr.Blocks:
    # theme/css 放在 Blocks：兼容当前 Gradio；launch() 传 theme 会 TypeError（与 Win/Mac 无关）
    with gr.Blocks(title="EchoSelf", theme=gr.themes.Soft(), css=CSS) as demo:

        gr.Markdown("# 🪞 EchoSelf\n**从聊天记录训练数字分身。**")
        if (_DEVICE_INFO.get("cuda_setup_hint") or "").strip():
            gr.Markdown(
                _DEVICE_INFO["cuda_setup_hint"],
                elem_classes=["cuda-setup-banner"],
            )

        # ── 全局系统提示词（一处修改，全局同步）──────────────────
        with gr.Accordion("⚙️ 全局系统提示词（所有步骤共用，修改后自动同步）", open=False):
            global_system_prompt = gr.Textbox(
                label="系统提示词",
                value="请你扮演一个真实的人，用自然的方式进行对话。",
                lines=2,
                info="此处填写一次，数据处理、模型训练、模型对话、Ollama 导出四个步骤将自动共用同一段系统提示词。",
            )

        with gr.Tabs():

            # ── Tab 1：数据处理 ────────────────────────────
            with gr.Tab("📦 数据处理"):

                gr.Markdown("### 第一步：指定聊天记录所在文件夹")

                with gr.Row():
                    folder_path_input = gr.Textbox(
                        label="聊天记录文件夹路径",
                        placeholder="点击右侧按钮选择，或直接输入绝对路径",
                        scale=5,
                        interactive=True,
                    )
                    pick_btn = gr.Button("📂 选择文件夹", variant="secondary", scale=1, min_width=120)

                with gr.Row():
                    my_id_input = gr.Textbox(
                        label="我的微信账号 ID",
                        placeholder="选择文件夹后自动识别，也可手动填写 wxid_xxxx",
                        scale=4,
                        interactive=True,
                        info="点击「选择文件夹」后会自动从聊天记录中识别",
                    )

                # 选择文件夹后显示识别摘要
                folder_hint_box = gr.Markdown()

                preview_btn = gr.Button("🔍 扫描文件夹 · 预览消息类型分布", variant="secondary", size="sm")
                type_stats_box = gr.Markdown()

                # 选择文件夹 → 填充路径 + 自动识别账号 ID
                pick_btn.click(
                    pick_folder,
                    inputs=[my_id_input],
                    outputs=[folder_path_input, my_id_input, folder_hint_box],
                )
                # 扫描按钮 → 消息类型分布 + 确认账号 ID
                preview_btn.click(
                    preview_type_stats,
                    inputs=[folder_path_input, my_id_input],
                    outputs=[type_stats_box, my_id_input],
                )

                gr.Markdown("### 第二步：配置处理参数")

                with gr.Row():
                    with gr.Column():
                        system_prompt_input = gr.Textbox(
                            label="系统提示词（训练时注入，与顶部全局设置同步）",
                            value="请你扮演一个真实的人，用自然的方式进行对话。",
                            lines=2,
                        )
                        with gr.Row():
                            time_window_input = gr.Slider(
                                1, 30, value=5, step=1,
                                label="QA 匹配时间窗口（分钟）",
                                info="问答两条消息时间间隔不超过此值才组成有效对话对",
                            )
                            combine_window_input = gr.Slider(
                                1, 10, value=2, step=1,
                                label="连续消息合并窗口（分钟）",
                                info="同一人在此时间内发送的多条消息合并为一条",
                            )
                        with gr.Row():
                            min_len_input = gr.Number(value=2, label="最短消息长度（字符）", precision=0)
                            max_len_input = gr.Number(value=500, label="最长消息长度（字符）", precision=0)

                    with gr.Column():
                        enable_pii_input = gr.Checkbox(
                            label="开启隐私脱敏（识别手机号、邮箱、身份证等）",
                            value=True,
                        )
                        blocked_words_input = gr.Textbox(
                            label="自定义禁用词（每行一个，包含此词的整条消息将被删除）",
                            placeholder="密码\n银行卡\n某人真实姓名",
                            lines=5,
                        )
                        output_format_input = gr.Radio(
                            choices=["alpaca", "sharegpt"],
                            value="alpaca",
                            label="输出格式",
                            info="alpaca：单轮指令；sharegpt：多轮对话（均兼容 LLaMA-Factory）",
                        )

                process_btn = gr.Button("🚀 开始处理数据", variant="primary", size="lg")

                with gr.Row():
                    stats_output = gr.Markdown(label="处理统计")
                    preview_output = gr.Markdown(label="数据预览")

                download_output = gr.File(label="下载训练数据")

                process_btn.click(
                    process_data,
                    inputs=[
                        folder_path_input, my_id_input,
                        time_window_input, combine_window_input,
                        min_len_input, max_len_input,
                        blocked_words_input, enable_pii_input,
                        system_prompt_input, output_format_input,
                    ],
                    outputs=[stats_output, preview_output, download_output],
                )

            # ── Tab 2：模型下载 ────────────────────────────
            with gr.Tab("⬇️ 模型下载"):

                gr.Markdown("### 下载预训练模型")
                gr.Markdown(
                    "国内用户推荐 **ModelScope**（无需代理，速度快）；"
                    "境外用户或需要最新版本可选 **HuggingFace**。"
                )

                # 下载工具状态
                _ms_status = "✅ modelscope 已安装" if _MS_OK else "❌ modelscope 未安装 → `pip install modelscope`"
                _hf_status = "✅ huggingface-hub 已安装" if _HF_OK else "❌ huggingface-hub 未安装 → `pip install huggingface-hub`"
                with gr.Row():
                    gr.Markdown(_ms_status)
                    gr.Markdown(_hf_status)

                gr.Markdown("---")

                with gr.Row():
                    dl_source = gr.Radio(
                        choices=["ModelScope（推荐·国内直连）", "HuggingFace"],
                        value="ModelScope（推荐·国内直连）",
                        label="下载源",
                        scale=2,
                    )
                    dl_model_choice = gr.Dropdown(
                        choices=_MODEL_CHOICES,
                        value=_DEFAULT_CHOICE,
                        label="选择预设模型",
                        info=(
                            f"当前设备：{_DEVICE_INFO['icon']} {_DEVICE_INFO['device']} — "
                            "标注 ✅ 的项已按本机显存/统一内存估算；⚠️ 表示可能超出容量，请谨慎选择"
                        ),
                        scale=4,
                    )

                with gr.Row():
                    dl_model_id = gr.Textbox(
                        label="模型 ID（自动填写，可手动修改）",
                        placeholder="Qwen/Qwen2.5-1.5B-Instruct",
                        scale=3,
                    )
                    dl_local_dir = gr.Textbox(
                        label="本地保存路径",
                        placeholder="./models/Qwen2.5-1.5B-Instruct",
                        scale=2,
                    )

                with gr.Row():
                    dl_start_btn = gr.Button("⬇️ 开始下载", variant="primary", size="lg")
                    dl_stop_btn  = gr.Button("⏹️ 停止",    variant="stop",    size="lg")

                dl_log_output = gr.Textbox(
                    label="下载日志",
                    lines=18,
                    max_lines=25,
                    autoscroll=True,
                    interactive=False,
                    elem_classes=["download-log"],
                )
                dl_stop_status = gr.Markdown()

                # 切换模型或下载源时自动更新 ID 和路径
                dl_model_choice.change(
                    fn=on_dl_model_change,
                    inputs=[dl_model_choice, dl_source],
                    outputs=[dl_model_id, dl_local_dir],
                )
                dl_source.change(
                    fn=on_dl_model_change,
                    inputs=[dl_model_choice, dl_source],
                    outputs=[dl_model_id, dl_local_dir],
                )

                dl_start_btn.click(
                    start_download,
                    inputs=[dl_source, dl_model_id, dl_local_dir],
                    outputs=[dl_log_output],
                )
                dl_stop_btn.click(stop_download, outputs=[dl_stop_status])

            # ── Tab 3：模型训练 ────────────────────────────
            with gr.Tab("🎯 模型训练") as tab_train:

                # 设备信息面板
                gr.Markdown(_device_banner(), elem_classes=["device-info"])

                # LLaMA-Factory 状态
                _ok, _path = check_llamafactory()
                lf_status = f"✅ LLaMA-Factory 已安装：`{_path}`" if _ok else "❌ 未安装 LLaMA-Factory，请运行：`pip install llamafactory`"
                gr.Markdown(lf_status)

                gr.Markdown("---")
                gr.Markdown("### 模型选择")

                # 扫描本地已下载的模型
                _local_models = scan_local_models()
                _local_default = _local_models[0] if _local_models else None
                _no_model_hint = (
                    "" if _local_models
                    else "⚠️ `./models/` 目录中暂无模型，请先在「⬇️ 模型下载」Tab 下载模型。"
                )

                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=_local_models,
                        value=_local_default,
                        label="本地模型",
                        info="显示 ./models/ 下已下载的模型，选择后对话模板会自动填写；点击「🔄 刷新」重新扫描目录",
                        allow_custom_value=True,
                        scale=3,
                    )
                    refresh_model_btn = gr.Button("🔄 刷新", variant="secondary", scale=1, min_width=80)
                    model_path_input = gr.Textbox(
                        label="自定义模型路径（留空则使用上方选中路径）",
                        info="若模型不在 ./models/ 目录下，可在此填写完整绝对路径，优先级高于下拉选择",
                        placeholder="/path/to/your/model",
                        scale=3,
                    )
                    template_input = gr.Textbox(
                        label="对话模板（自动填写）",
                        info="不同模型有固定的对话格式模板，例如 Qwen 用 qwen、Llama 用 llama3。选择模型后自动匹配，通常无需手动修改",
                        value=get_template_for_model(_local_default or ""),
                        scale=1,
                    )

                no_model_hint = gr.Markdown(_no_model_hint)

                gr.Markdown("### 数据 & 输出")
                with gr.Row():
                    dataset_path_input = gr.Textbox(
                        label="训练数据路径",
                        info="指向「数据处理」Tab 生成的 sft_data.json 文件路径，这是模型学习的原材料",
                        value=str(DATASET_PATH),
                        scale=3,
                    )
                    output_dir_input = gr.Textbox(
                        label="模型输出目录",
                        info="训练完成后 LoRA 适配器权重（adapter_model.bin / adapter_config.json 等）会保存在此目录，可在「💬 模型对话」Tab 中加载",
                        value="./output/model",
                        scale=2,
                    )

                train_system_input = gr.Textbox(
                    label="系统提示词（System Prompt，与顶部全局设置同步）",
                    info="每条训练样本开头都会注入这段话，告诉模型「你是谁」。训练时和推理时的系统提示词保持一致，效果最好。也可在顶部「全局系统提示词」统一修改。",
                    value="请你扮演一个真实的人，用自然的方式进行对话。",
                    lines=2,
                )

                with gr.Accordion("📚 微调入门 — 参数解释（点击展开）", open=False):
                    gr.Markdown("""
### 什么是 LoRA 微调？

大语言模型有几十亿个参数，直接全量训练需要几十 GB 显存。**LoRA（Low-Rank Adaptation）** 是一种高效的微调方法：
- 不修改原始模型权重，而是在特定层旁边插入两个极小的矩阵（A 和 B）
- 训练时只更新这两个小矩阵（参数量不到原模型的 1%）
- 推理时将小矩阵合并回原模型，效果接近全量微调

**类比**：原模型是一本教科书，LoRA 就是在书边贴便利贴——不改书本，只加注释。

---

### 参数详解

| 参数 | 作用 | 建议范围 |
|------|------|---------|
| **LoRA Rank** | 插入矩阵的"维度"，越大学习能力越强，显存占用也越大 | 个人风格：4~16；复杂任务：32~64 |
| **训练轮数 (Epochs)** | 数据集被完整训练几遍。轮数过少欠拟合，过多过拟合 | 1000条数据以下：3~5轮；大数据集：1~2轮 |
| **Batch Size** | 每次同时送入多少条样本。Mac/MPS 建议设 1，GPU 可设 2~8 | Mac: 1；NVIDIA 16GB: 2~4 |
| **梯度累积步数** | 等效扩大 Batch Size。实际等效 batch = Batch × 梯度累积。可在显存有限时模拟大 batch | 8~16（配合 Batch=1 使用） |
| **学习率** | 每步更新参数的步长。太大会震荡，太小收敛慢 | 1e-4（常用）；2e-4（更快但可能不稳定） |
| **最大序列长度** | 每条训练样本被截断的最大 token 数。越长显存占用越大 | Mac 建议 512；NVIDIA GPU 可用 1024~2048 |

---

### 如何判断训练好不好？

- **Loss 持续下降** → 模型在正常学习 🟢
- **Loss 下降后趋于平稳** → 收敛，可以停止或适当增加轮数 🟡
- **Loss 反弹上升** → 过拟合或学习率过大，建议降低学习率 🔴
- **Loss 一直不降** → 数据量太少、学习率太小或数据质量差，检查数据

---

### 什么是过拟合？

模型记住了训练数据，但遇到新问题时表现很差。就像学生死记答案，换个题型就不会了。
- **表现**：训练 Loss 很低，但对话时生硬、重复、只会复述训练集的句子
- **解决**：减少训练轮数、增加数据多样性、适当降低 LoRA Rank

---

### SFT 是什么？

**SFT（Supervised Fine-Tuning，监督微调）** 是本项目使用的训练方式：
- 给模型提供（问题，回答）对，让它学习如何在给定问题下给出正确回答
- 就像用示例教学生：「遇到这种问题，应该这样回答」
- 与强化学习（RLHF）不同，SFT 不需要打分模型，更简单高效
""")

                gr.Markdown("### 训练超参")
                with gr.Row():
                    lora_rank_input    = gr.Slider(4, 64, value=8,   step=4,   label="LoRA Rank",
                        info="LoRA 矩阵维度。越大模型学习能力越强，但显存占用也越多。个人风格模仿推荐 8~16；显存/统一内存紧张时请适当降低")
                    epochs_input       = gr.Slider(1, 10, value=3,   step=0.5, label="训练轮数（Epochs）",
                        info="整个数据集被训练几遍。数据少（<1000条）可设 3~5 轮；数据多（>5000条）设 1~2 轮即可，过多会过拟合")
                    batch_size_input   = gr.Slider(1, 8,  value=1,   step=1,   label="Batch Size",
                        info="每步同时处理的样本数。Mac/MPS 必须设为 1，否则易内存溢出；NVIDIA 显卡可适当增大")
                    grad_accum_input   = gr.Slider(1, 32, value=8,   step=1,   label="梯度累积步数",
                        info="等效增大 Batch Size 而不增加显存。等效 Batch = Batch Size × 梯度累积步数。配合 Batch=1 使用时，设为 8 等效于 Batch=8")
                with gr.Row():
                    lr_input           = gr.Number(value=1e-4, label="学习率（Learning Rate）",
                        info="控制每步参数更新的幅度。推荐 1e-4（即 0.0001）。太大（>5e-4）容易震荡，太小（<1e-5）收敛极慢")
                    cutoff_input       = gr.Slider(128, 2048, value=512, step=128, label="最大序列长度（Token）",
                        info="每条样本超过此长度会被截断。显存/统一内存一般可设 512；容量充足时可试 1024~2048。越长越吃内存")

                gr.Markdown(
                    f"> **精度设置由设备自动决定**：当前设备为 **{_DEVICE_INFO['device']}**，"
                    f"训练时将自动使用 {'bf16' if _DEVICE_INFO['use_bf16'] else 'fp16' if _DEVICE_INFO['use_fp16'] else 'fp32'}，"
                    f"Flash Attention：{_DEVICE_INFO['flash_attn']}。"
                )

                # ── 断点续训区 ─────────────────────────────────
                gr.Markdown("### 断点续训")
                with gr.Row():
                    resume_checkbox = gr.Checkbox(
                        label="从断点继续训练",
                        value=False,
                        info="勾选后将从已保存的 checkpoint 继续训练，不会覆盖已有进度。训练中断后可使用此功能恢复。",
                        scale=1,
                    )
                    _init_ckpts = scan_checkpoints("./output/model")
                    resume_checkpoint_input = gr.Dropdown(
                        choices=_init_ckpts,
                        value=_init_ckpts[0] if _init_ckpts else None,
                        label="选择断点（checkpoint）",
                        info="列出训练输出目录中已保存的 checkpoint，按步数从大到小排序，通常选最新（第一个）即可",
                        allow_custom_value=True,
                        interactive=True,
                        scale=3,
                        visible=bool(_init_ckpts),
                    )
                    refresh_ckpt_btn = gr.Button("🔄 扫描断点", variant="secondary", scale=1, min_width=90)

                resume_hint = gr.Markdown(
                    "⚠️ 未发现已保存的 checkpoint，请先开始一次训练（每 100 步自动保存一次）。" if not _init_ckpts else "",
                    visible=not bool(_init_ckpts),
                )

                def _toggle_resume(checked: bool):
                    """勾选续训时显示 checkpoint 选择器"""
                    ckpts = scan_checkpoints("./output/model")
                    has = bool(ckpts)
                    return (
                        gr.update(choices=ckpts, value=ckpts[0] if has else None, visible=checked and has),
                        gr.update(visible=checked and not has),
                    )

                def _refresh_checkpoints(output_dir: str):
                    """按输出目录重新扫描 checkpoint"""
                    ckpts = scan_checkpoints(output_dir or "./output/model")
                    has = bool(ckpts)
                    return (
                        gr.update(choices=ckpts, value=ckpts[0] if has else None, visible=has),
                        gr.update(visible=not has, value="⚠️ 未发现 checkpoint，请检查输出目录。" if not has else ""),
                    )

                resume_checkbox.change(
                    _toggle_resume,
                    inputs=[resume_checkbox],
                    outputs=[resume_checkpoint_input, resume_hint],
                )
                refresh_ckpt_btn.click(
                    _refresh_checkpoints,
                    inputs=[output_dir_input],
                    outputs=[resume_checkpoint_input, resume_hint],
                )
                # 切换输出目录时自动刷新可用 checkpoint
                output_dir_input.change(
                    _refresh_checkpoints,
                    inputs=[output_dir_input],
                    outputs=[resume_checkpoint_input, resume_hint],
                )

                train_inputs = [
                    model_choice, model_path_input, template_input,
                    dataset_path_input, output_dir_input, train_system_input,
                    lora_rank_input, epochs_input, batch_size_input,
                    grad_accum_input, lr_input, cutoff_input,
                    resume_checkbox, resume_checkpoint_input,
                ]

                # 选择模型时自动更新模板
                model_choice.change(
                    fn=lambda c: get_template_for_model(c or ""),
                    inputs=[model_choice],
                    outputs=[template_input],
                )

                def _refresh_models():
                    """重新扫描本地模型目录，更新下拉列表"""
                    models = scan_local_models()
                    hint = (
                        "" if models
                        else "⚠️ `./models/` 目录中暂无模型，请先在「⬇️ 模型下载」Tab 下载模型。"
                    )
                    new_val = models[0] if models else None
                    return gr.update(choices=models, value=new_val), hint

                refresh_model_btn.click(
                    fn=_refresh_models,
                    outputs=[model_choice, no_model_hint],
                )

                cmd_preview_btn = gr.Button("👀 预览训练命令", variant="secondary")
                cmd_preview_output = gr.Markdown()

                with gr.Row():
                    start_btn = gr.Button("▶️ 开始训练", variant="primary", size="lg")
                    stop_btn  = gr.Button("⏹️ 停止训练", variant="stop",    size="lg")
                    restore_btn = gr.Button("📋 恢复上次训练记录", variant="secondary", size="lg")

                # 实时训练进度面板
                progress_output = gr.Markdown(
                    "_点击「▶️ 开始训练」后显示实时进度；刷新页面后点「📋 恢复上次训练记录」找回日志_",
                    elem_classes=["train-progress"],
                )

                # Loss 曲线图
                loss_plot_output = gr.Plot(label="Loss 曲线（每 5 步更新一次）")

                train_log_output = gr.Textbox(
                    label="训练日志（原始输出）",
                    lines=15,
                    max_lines=22,
                    autoscroll=True,
                    interactive=False,
                )
                stop_status = gr.Markdown()

                # model_choice 现在直接是本地路径，自定义路径优先
                def _resolve_model(choice: str, custom_path: str) -> str:
                    return custom_path.strip() if custom_path.strip() else (choice or "")

                def _restore_training_log(out_dir: str):
                    """
                    刷新页面后恢复上次训练记录：
                    1. 读取持久化日志文件显示原始日志
                    2. 解析 trainer_log.jsonl 重建 Loss 曲线
                    """
                    # 读取原始日志
                    log_text = ""
                    if TRAIN_LOG_FILE.exists():
                        try:
                            log_text = TRAIN_LOG_FILE.read_text(encoding="utf-8")
                        except Exception:
                            log_text = "⚠️ 读取日志文件失败"
                    else:
                        log_text = "⚠️ 未找到日志文件，请先开始一次训练"

                    # 解析 trainer_log.jsonl 恢复 Loss 曲线
                    import json as _json
                    history: list[dict] = []
                    jsonl_path = Path(out_dir or "./output/model") / "trainer_log.jsonl"
                    if jsonl_path.exists():
                        try:
                            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                                line = line.strip()
                                if not line:
                                    continue
                                rec = _json.loads(line)
                                step = rec.get("current_steps")
                                loss = rec.get("loss")
                                if step and loss:
                                    history.append({"step": step, "loss": loss})
                        except Exception:
                            pass

                    plot = _make_loss_plot(history) if history else None
                    progress = ""
                    if history:
                        last = history[-1]
                        progress = (
                            f"**📋 已从历史记录恢复**\n\n"
                            f"- 最近步数：**{last['step']}**\n"
                            f"- 最近 Loss：**{last['loss']:.4f}**\n"
                            f"- 共记录 **{len(history)}** 个数据点\n\n"
                            f"> 如训练仍在进行，点击「▶️ 开始训练」会重新开始，请使用断点续训功能"
                        )
                    else:
                        progress = "⚠️ 未找到 `trainer_log.jsonl`，Loss 曲线无法恢复"

                    return log_text, progress, plot

                def _wrap_start(*args):
                    resolved = _resolve_model(args[0], args[1])
                    yield from start_training(resolved, *args[1:])

                def _wrap_preview(*args):
                    # preview 不需要 resume 参数，只取前 12 个
                    resolved = _resolve_model(args[0], args[1])
                    return get_command_preview(resolved, *args[1:12])

                cmd_preview_btn.click(_wrap_preview, inputs=train_inputs, outputs=[cmd_preview_output])
                start_btn.click(
                    _wrap_start, inputs=train_inputs,
                    outputs=[train_log_output, progress_output, loss_plot_output],
                )
                stop_btn.click(stop_training, outputs=[stop_status])
                restore_btn.click(
                    _restore_training_log,
                    inputs=[output_dir_input],
                    outputs=[train_log_output, progress_output, loss_plot_output],
                )

            # ── Tab 4：模型对话 ────────────────────────────
            with gr.Tab("💬 模型对话") as tab_inf:

                gr.Markdown("### 与训练好的数字分身对话")
                gr.Markdown(
                    "选择基础模型和 LoRA adapter，加载后即可开始对话。"
                    "首次加载约需 20~60 秒，请耐心等待。"
                )

                # 模型加载区
                with gr.Row():
                    inf_base_model = gr.Dropdown(
                        choices=scan_local_models(),
                        value=None,
                        label="基础模型（./models/ 下已下载）",
                        allow_custom_value=True,
                        scale=3,
                    )
                    inf_adapter = gr.Dropdown(
                        choices=scan_local_adapters(),
                        value=None,
                        label="LoRA Adapter（./output/ 下训练产物，可不选）",
                        allow_custom_value=True,
                        scale=3,
                    )
                    inf_refresh_btn = gr.Button("🔄 刷新列表", variant="secondary", scale=1, min_width=90)

                with gr.Row():
                    inf_load_btn   = gr.Button("🚀 加载模型", variant="primary",   size="lg", scale=2)
                    inf_unload_btn = gr.Button("🗑️ 卸载模型", variant="secondary", size="lg", scale=1)

                inf_status = gr.Markdown(
                    "_尚未加载模型_",
                    elem_classes=["inference-status"],
                )

                gr.Markdown("---")

                # 对话参数
                with gr.Row():
                    inf_system = gr.Textbox(
                        label="系统提示词（与顶部全局设置同步）",
                        value="请你扮演一个真实的人，用自然的方式进行对话。",
                        scale=4,
                        lines=1,
                    )
                    inf_temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                        label="Temperature（越高越随机）", scale=2)
                    inf_max_tokens = gr.Slider(64, 1024, value=256, step=32,
                                               label="最大生成长度", scale=2)

                # 聊天界面
                inf_chatbot = gr.Chatbot(
                    label="对话",
                    height=420,
                    type="messages",      # 使用 openai 风格 {role, content} 格式，与 _chat_fn 返回一致
                    allow_tags=False,     # 消除 Gradio 5.x → 6.0 的 DeprecationWarning
                )
                with gr.Row():
                    inf_input = gr.Textbox(
                        placeholder="输入消息，按 Enter 发送…",
                        label="",
                        scale=5,
                        lines=1,
                    )
                    inf_send_btn  = gr.Button("发送 ↵",  variant="primary",   scale=1)
                    inf_clear_btn = gr.Button("清空对话", variant="secondary", scale=1)

                # ── 事件绑定 ──

                def _refresh_inf_lists():
                    return (
                        gr.update(choices=scan_local_models()),
                        gr.update(choices=scan_local_adapters()),
                    )

                inf_refresh_btn.click(_refresh_inf_lists, outputs=[inf_base_model, inf_adapter])

                def _load_model_ui(base, adapter):
                    for msg in load_model(base or "", adapter or ""):
                        yield msg

                def _unload_model_ui():
                    return unload_model()

                inf_load_btn.click(
                    _load_model_ui,
                    inputs=[inf_base_model, inf_adapter],
                    outputs=[inf_status],
                )
                inf_unload_btn.click(_unload_model_ui, outputs=[inf_status])

                def _chat_fn(message, history, system, temperature, max_tokens):
                    # Gradio 6+ Chatbot 要求 history 为 [{role, content}, ...]，不再使用 [user, bot] 元组
                    if not message.strip():
                        yield "", history or []
                        return
                    normalized: list[dict] = []
                    for m in history or []:
                        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                            normalized.append(
                                {"role": m["role"], "content": m.get("content") or ""}
                            )
                        elif isinstance(m, (list, tuple)) and len(m) >= 2:
                            if m[0]:
                                normalized.append({"role": "user", "content": str(m[0])})
                            if m[1]:
                                normalized.append({"role": "assistant", "content": str(m[1])})
                    history = normalized
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": ""})
                    yield "", history
                    partial = ""
                    prior = history[:-2]
                    for chunk in chat_stream(
                        message, prior, system, temperature, max_tokens
                    ):
                        partial = chunk
                        history[-1]["content"] = partial
                        yield "", history
                    yield "", history

                inf_send_btn.click(
                    _chat_fn,
                    inputs=[inf_input, inf_chatbot, inf_system, inf_temp, inf_max_tokens],
                    outputs=[inf_input, inf_chatbot],
                )
                inf_input.submit(
                    _chat_fn,
                    inputs=[inf_input, inf_chatbot, inf_system, inf_temp, inf_max_tokens],
                    outputs=[inf_input, inf_chatbot],
                )
                inf_clear_btn.click(lambda: ([], ""), outputs=[inf_chatbot, inf_input])

            # ── Tab 5：模型导出 ────────────────────────────
            with gr.Tab("📤 模型导出"):

                gr.Markdown("### 将训练好的模型导出为可用格式")
                gr.Markdown(
                    "按顺序完成三个步骤：**合并 LoRA → 转换 GGUF → 导入 Ollama**。\n"
                    "也可以只做 Step 1 得到完整 HuggingFace 模型，直接用于 transformers 推理。"
                )

                # ── 公共：环境检测 ──────────────────────────
                _llama_ok, _llama_method, _llama_path = check_llama_cpp()
                _ollama_ok, _ = check_ollama()
                _env_lines = []
                _env_lines.append(
                    f"- llama.cpp 转换工具：{'✅ 已安装（' + _llama_method + '）' if _llama_ok else '❌ 未安装 — `brew install llama.cpp`'}"
                )
                _env_lines.append(
                    f"- Ollama：{'✅ 已安装' if _ollama_ok else '❌ 未安装 — [ollama.com/download](https://ollama.com/download)'}"
                )
                gr.Markdown("\n".join(_env_lines))
                gr.Markdown("---")

                # ── Step 1：合并 LoRA ────────────────────────
                gr.Markdown("## Step 1 — 合并 LoRA Adapter → 完整模型")
                gr.Markdown(
                    "> 将基础模型 + LoRA adapter 合并为一个独立的完整模型，"
                    "保存为 HuggingFace safetensors 格式，可直接用于推理或继续导出。"
                )

                with gr.Row():
                    exp_base_model = gr.Dropdown(
                        choices=scan_local_models(),
                        value=None,
                        label="基础模型",
                        info="选择训练时使用的基础模型",
                        allow_custom_value=True,
                        scale=3,
                    )
                    exp_adapter = gr.Dropdown(
                        choices=scan_local_adapters(),
                        value=None,
                        label="LoRA Adapter",
                        info="选择训练输出的 adapter 目录（含 adapter_config.json）",
                        allow_custom_value=True,
                        scale=3,
                    )
                    exp_refresh_btn = gr.Button("🔄 刷新", variant="secondary", scale=1, min_width=80)

                with gr.Row():
                    exp_template = gr.Textbox(
                        label="对话模板",
                        value="qwen",
                        info="与训练时保持一致，Qwen 系列填 qwen，Llama 填 llama3",
                        scale=1,
                    )
                    exp_merge_output = gr.Textbox(
                        label="合并后模型保存目录",
                        value="./output/merged_model",
                        info="合并完整权重的保存位置，约占 2~8 GB 磁盘空间",
                        scale=3,
                    )

                with gr.Row():
                    merge_btn  = gr.Button("🔗 开始合并", variant="primary", scale=2)
                    merge_stop = gr.Button("⏹️ 停止", variant="stop", scale=1)

                merge_log = gr.Textbox(
                    label="合并日志",
                    lines=8, max_lines=12,
                    autoscroll=True, interactive=False,
                )
                gr.Markdown("---")

                # ── Step 2：转换 GGUF ───────────────────────
                gr.Markdown("## Step 2 — 转换 GGUF 格式（Ollama / llama.cpp）")
                gr.Markdown(
                    "> GGUF 是 llama.cpp 的专用格式，Ollama 使用此格式运行模型。"
                    "Q4_K_M 量化后 3B 模型约 2 GB，推荐首选。"
                )

                with gr.Row():
                    gguf_merged_dir = gr.Textbox(
                        label="已合并模型目录（Step 1 输出）",
                        value="./output/merged_model",
                        info="填入 Step 1 的输出目录，或任意已合并的 HuggingFace 模型目录",
                        scale=3,
                    )
                    gguf_quant = gr.Dropdown(
                        choices=[f"{q[0]} — {q[1]}" for q in QUANT_OPTIONS],
                        value=f"{QUANT_OPTIONS[0][0]} — {QUANT_OPTIONS[0][1]}",
                        label="量化精度",
                        info="影响模型体积和推理质量，Q4_K_M 是最佳平衡点",
                        scale=2,
                    )
                    gguf_output = gr.Textbox(
                        label="GGUF 输出路径",
                        value="./output/model.gguf",
                        info="生成的 .gguf 文件路径",
                        scale=2,
                    )

                if not _llama_ok:
                    gr.Markdown(
                        "> ⚠️ **未检测到 llama.cpp**，转换功能不可用。\n"
                        "> 请先安装：`brew install llama.cpp`，安装后重启应用。"
                    )

                with gr.Row():
                    gguf_btn  = gr.Button("⚙️ 开始转换 GGUF", variant="primary", scale=2,
                                          interactive=_llama_ok)
                    gguf_stop = gr.Button("⏹️ 停止", variant="stop", scale=1)

                gguf_log = gr.Textbox(
                    label="转换日志",
                    lines=8, max_lines=12,
                    autoscroll=True, interactive=False,
                )
                gr.Markdown("---")

                # ── Step 3：导入 Ollama ──────────────────────
                gr.Markdown("## Step 3 — 导入 Ollama")
                gr.Markdown(
                    "> 生成 Modelfile 后一键导入 Ollama，之后可在终端用 `ollama run 模型名` 对话，"
                    "或接入 Open WebUI、ChatBox 等客户端。"
                )

                with gr.Row():
                    ollama_gguf = gr.Textbox(
                        label="GGUF 文件路径（Step 2 输出）",
                        value="./output/model.gguf",
                        scale=3,
                    )
                    ollama_name = gr.Textbox(
                        label="Ollama 模型名称",
                        value="echoself",
                        info="在 Ollama 中显示的名称，例如 echoself 或 my-qwen",
                        scale=2,
                    )
                ollama_system = gr.Textbox(
                    label="系统提示词（注入到 Modelfile，与顶部全局设置同步）",
                    value="请你扮演一个真实的人，用自然的方式进行对话。",
                    lines=2,
                )

                modelfile_preview = gr.Code(
                    label="Modelfile 预览",
                    language=None,
                    lines=8,
                    interactive=False,
                    visible=False,
                )

                if not _ollama_ok:
                    gr.Markdown(
                        "> ⚠️ **未检测到 Ollama**，导入功能不可用。\n"
                        "> 请先安装：[ollama.com/download](https://ollama.com/download)，安装后重启应用。"
                    )

                with gr.Row():
                    modelfile_btn = gr.Button("📄 生成 Modelfile", variant="secondary", scale=1)
                    ollama_btn    = gr.Button("🚀 一键导入 Ollama", variant="primary", scale=2,
                                             interactive=_ollama_ok)
                    ollama_stop   = gr.Button("⏹️ 停止", variant="stop", scale=1)

                ollama_log = gr.Textbox(
                    label="导入日志",
                    lines=6, max_lines=10,
                    autoscroll=True, interactive=False,
                )

                # ── 事件绑定 ────────────────────────────────

                def _exp_refresh():
                    return (
                        gr.update(choices=scan_local_models()),
                        gr.update(choices=scan_local_adapters()),
                    )
                exp_refresh_btn.click(_exp_refresh, outputs=[exp_base_model, exp_adapter])

                # 选择基础模型时自动填模板
                exp_base_model.change(
                    fn=lambda c: get_template_for_model(c or ""),
                    inputs=[exp_base_model],
                    outputs=[exp_template],
                )

                # 合并
                _merge_logs: list[str] = []

                def _start_merge(base, adapter, template, out_dir):
                    global _merge_logs
                    _merge_logs = []
                    if _merge_process.is_running:
                        yield "⚠️ 合并进程仍在运行，请先停止"
                        return
                    done: dict = {"done": False, "code": -1}
                    def on_log(l): _merge_logs.append(l)
                    def on_done(c):
                        done["done"] = True
                        done["code"] = c
                    ok = _merge_process.start(base or "", adapter or "", template or "qwen", out_dir or "./output/merged_model", on_log, on_done)
                    if not ok:
                        yield "\n".join(_merge_logs)
                        return
                    import time
                    while not done["done"]:
                        time.sleep(0.5)
                        yield "\n".join(_merge_logs[-100:])
                    status = "✅ 合并完成！" if done["code"] == 0 else f"❌ 合并失败（code={done['code']}）"
                    _merge_logs.append(f"\n{status}")
                    yield "\n".join(_merge_logs[-100:])

                merge_btn.click(
                    _start_merge,
                    inputs=[exp_base_model, exp_adapter, exp_template, exp_merge_output],
                    outputs=[merge_log],
                )
                merge_stop.click(lambda: _merge_process.stop() or "⏹ 已发送停止信号")

                # GGUF 转换
                _gguf_logs: list[str] = []

                def _start_gguf(merged_dir, quant_str, output_path):
                    global _gguf_logs
                    _gguf_logs = []
                    if _gguf_process.is_running:
                        yield "⚠️ 转换进程仍在运行，请先停止"
                        return
                    # 从下拉选项中提取量化类型（格式: "Q4_K_M — ..."）
                    quant = quant_str.split(" — ")[0].strip() if quant_str else "Q4_K_M"
                    done: dict = {"done": False, "code": -1}
                    def on_log(l): _gguf_logs.append(l)
                    def on_done(c):
                        done["done"] = True
                        done["code"] = c
                    ok = _gguf_process.start(merged_dir or "", output_path or "./output/model.gguf", quant, on_log, on_done)
                    if not ok:
                        yield "\n".join(_gguf_logs)
                        return
                    import time
                    while not done["done"]:
                        time.sleep(0.5)
                        yield "\n".join(_gguf_logs[-100:])
                    yield "\n".join(_gguf_logs[-100:])

                gguf_btn.click(
                    _start_gguf,
                    inputs=[gguf_merged_dir, gguf_quant, gguf_output],
                    outputs=[gguf_log],
                )
                gguf_stop.click(lambda: _gguf_process.stop() or "⏹ 已发送停止信号")

                # 生成 Modelfile
                def _gen_modelfile(gguf_path, name, system):
                    content = generate_modelfile(
                        gguf_path or "./output/model.gguf",
                        name or "echoself",
                        system or "",
                    )
                    mf_path = str((Path(gguf_path or "./output/model.gguf").parent / "Modelfile").resolve())
                    save_modelfile(content, mf_path)
                    return gr.update(value=content, visible=True), f"✅ Modelfile 已保存：{mf_path}"

                modelfile_btn.click(
                    _gen_modelfile,
                    inputs=[ollama_gguf, ollama_name, ollama_system],
                    outputs=[modelfile_preview, ollama_log],
                )

                # 导入 Ollama
                _ollama_logs: list[str] = []

                def _start_ollama(gguf_path, name, system):
                    global _ollama_logs
                    _ollama_logs = []
                    if _ollama_process.is_running:
                        yield "⚠️ 导入进程仍在运行，请稍候"
                        return
                    mf_path = str((Path(gguf_path or "./output/model.gguf").parent / "Modelfile").resolve())
                    # 若 Modelfile 不存在先自动生成
                    if not Path(mf_path).exists():
                        content = generate_modelfile(gguf_path or "./output/model.gguf", name or "echoself", system or "")
                        save_modelfile(content, mf_path)
                        _ollama_logs.append(f"📄 已自动生成 Modelfile：{mf_path}\n")
                    done: dict = {"done": False, "code": -1}
                    def on_log(l): _ollama_logs.append(l)
                    def on_done(c):
                        done["done"] = True
                        done["code"] = c
                    ok = _ollama_process.start(name or "echoself", mf_path, on_log, on_done)
                    if not ok:
                        yield "\n".join(_ollama_logs)
                        return
                    import time
                    while not done["done"]:
                        time.sleep(0.5)
                        yield "\n".join(_ollama_logs[-100:])
                    yield "\n".join(_ollama_logs[-100:])

                ollama_btn.click(
                    _start_ollama,
                    inputs=[ollama_gguf, ollama_name, ollama_system],
                    outputs=[ollama_log],
                )
                ollama_stop.click(lambda: _ollama_process.stop() or "⏹ 已发送停止信号")

                # 切换到导出 Tab 时自动刷新列表
            # ── Tab 5：帮助文档 ────────────────────────────
            with gr.Tab("📖 帮助文档"):
                gr.Markdown("""
<div class="help-content">

## EchoSelf 使用指南

### 📋 完整使用流程

**第一步** 导出聊天记录
> 使用 WeFlow 等工具将微信聊天记录导出为 `.json` 格式，所有文件放在同一文件夹。

**第二步** 数据处理
> 进入「📦 数据处理」Tab → 点击「📂 选择文件夹」（账号 ID 自动识别）→ 配置参数 → 「🚀 开始处理数据」

**第三步** 下载基础模型
> 进入「⬇️ 模型下载」Tab → 选择 ModelScope（国内推荐） → 选择模型 → 「⬇️ 开始下载」

**第四步** 启动训练
> 进入「🎯 模型训练」Tab → 点击「🔄 刷新」选择模型 → 配置超参 → 「▶️ 开始训练」

**第五步** 与分身对话
> 进入「💬 模型对话」Tab → 选择基础模型 + LoRA adapter → 「🚀 加载模型」→ 开始对话

---

### 🔤 技术名词解释

#### 基础模型 vs 微调模型
- **基础模型**：由 Qwen、LLaMA 等团队在大规模语料上预训练的通用语言模型，具备基础对话能力
- **微调模型（Fine-tuned Model）**：在基础模型上，用你自己的聊天记录进行额外训练，使其更像你

#### SFT（监督微调）
Supervised Fine-Tuning。用「问题 → 答案」格式的数据对基础模型进行有监督训练，让模型学会特定风格和内容。EchoSelf 生成的 QA 对就是用于 SFT 的训练数据。

#### LoRA（低秩适配）
Low-Rank Adaptation。一种高效微调技术，不直接修改原始模型的数百亿参数，而是只训练少量的「适配层」（约 0.07% 的参数），极大降低了显存需求和训练时间，同时效果接近全参数微调。

> 可以把 LoRA 理解为：给基础模型贴了一张「个性化贴纸」，贴纸很小但能显著改变模型行为。

#### QA 对（问答对）
EchoSelf 从聊天记录中提取的「对方说的话 → 你的回复」数据对。这是训练数据的基本单元，每条 QA 对教会模型在收到某类问题时如何回复。

#### Epoch（轮次）
遍历完整个训练数据集一次称为一个 Epoch。训练 3 Epochs 意味着每条数据被学习 3 遍。通常 1~5 轮即可，太多会过拟合。

#### Loss（损失值）
衡量模型预测与真实答案之间的差距，**Loss 越低 = 模型越接近你的说话风格**。训练过程中 Loss 应持续下降。典型初始值 2.0~3.0，收敛后约 0.5~1.5。

#### 学习率（Learning Rate）
控制每次参数更新的步长大小。太高 → Loss 震荡、发散；太低 → 训练极慢。常用起点为 `1e-4`（即 0.0001）。

#### Batch Size（批量大小）
每次梯度更新使用的样本数。显存/统一内存有限时建议设为 `1`，通过梯度累积等效增大有效批量。

#### 梯度累积（Gradient Accumulation）
解决显存不足问题的技巧。设置为 `8` 时，等效于 Batch Size × 8 = 8 条数据更新一次参数，但实际每次只处理 1 条，显存占用不增加。

#### bf16（BFloat16）
16 位浮点数格式。Apple Silicon 原生支持 bf16 但不支持 fp16，EchoSelf 会自动检测并切换，无需手动设置。

#### MPS（Metal Performance Shaders）
Apple 为 M 系列芯片提供的 GPU 加速框架，PyTorch 通过 MPS 利用 Apple Silicon 的 GPU 核心加速训练，相比纯 CPU 快 3~10 倍。

#### CUDA 是什么？和「有显卡」有什么区别？
- **CUDA** 是 NVIDIA 的并行计算平台；要在 PyTorch 里用 NVIDIA 显卡做训练，通常需要安装 **带 CUDA 的 PyTorch 构建**（以及与之匹配的显卡驱动）。
- **仅有显卡硬件**不等于已启用 CUDA：若 `pip install torch` 装到的是 **CPU 版**，`torch.cuda.is_available()` 仍为 `False`，训练会走 CPU。
- **建议步骤**：安装/更新 [NVIDIA 驱动](https://www.nvidia.cn/drivers/) → 打开 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择 **CUDA** 版本并复制给出的 `pip` 命令安装 → 终端验证：`python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。
- EchoSelf 会通过 `nvidia-smi` 检测 NVIDIA 显卡；若检测到显卡但当前环境未启用 CUDA，**界面顶部**会显示「请启用 CUDA」说明，按步骤操作后**重启 EchoSelf** 即可。

#### Grad Norm（梯度范数）
所有参数梯度的综合大小。数值稳定（0.1~5.0 之间）说明训练健康；异常大（>100）通常意味着学习率过高。

---

### 📊 训练质量评判标准

#### Loss 解读参考

| Loss 范围 | 含义 |
|-----------|------|
| 2.0 以上 | 训练初期，模型尚未适应数据 |
| 1.0 ~ 2.0 | 模型正在学习，有明显改善 |
| 0.5 ~ 1.0 | 训练效果良好，风格基本形成 |
| 0.5 以下 | 高度拟合，注意可能过拟合 |

#### 理想 Loss 曲线特征
- ✅ **持续下降**：健康状态，继续训练
- ✅ **平滑收敛**：Loss 下降后趋于平稳，表明训练完成
- ⚠️ **剧烈震荡**：学习率可能过高，建议降低至 `5e-5`
- ⚠️ **下降后回升**：过拟合信号，建议提前停止或减少 Epochs
- ❌ **持续上升**：配置有误，立即停止检查数据和参数

#### 过拟合 vs 欠拟合

**过拟合（Overfitting）**：模型把训练数据「背」下来了，只会重复原句，缺乏泛化能力。
- 表现：训练 Loss 极低（< 0.3），但对话时复读或语言僵化
- 解决：减少 Epochs（2 轮以内）、减小 LoRA Rank、增加数据量

**欠拟合（Underfitting）**：模型没学到你的风格。
- 表现：Loss 居高不下（> 2.0），输出仍像通用助手
- 解决：增加 Epochs、提高学习率、增加 QA 对数量

---

### ⚙️ 入门参考参数（显存/统一内存较紧时）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 模型 | 见上方「选择预设模型」中带 ✅ 的项 | 已按本机容量估算 |
| LoRA Rank | 8 | 显存紧张时优先保持 8 |
| Epochs | 3 | 避免过拟合 |
| Batch Size | 1 | 笔记本/统一内存常需为 1 |
| 梯度累积 | 8 | 等效 Batch 8 |
| 学习率 | 1e-4 | 标准起点 |
| 序列长度 | 512 | 平衡内存与上下文 |
| 精度 | bf16 / fp16 | 由设备自动设置 |

> **训练时间参考**：同等数据量下，Apple Silicon 笔记本约 2~4 小时、独显工作站通常更快（因机型与数据量差异大，仅供参考）。

---

### ❓ 常见问题

**Q：为什么 Loss 一直是 0？**
> 可能数据集为空，请检查「数据处理」步骤是否生成了有效 QA 对（需要 > 0 条）。

**Q：训练中断后能继续吗？**
> 暂不支持断点续训。如果训练中断，需重新开始，可在「输出目录」中看到部分保存的 checkpoint。

**Q：训练完成后模型保存在哪里？**
> LoRA adapter 保存在「模型输出目录」（默认 `./output/model/`），包含 `adapter_config.json` 和权重文件。

**Q：对话时模型不像我怎么办？**
> 1. 检查 QA 对数量（建议 > 1000 条）；2. 确认账号 ID 识别正确（确保数据是你说的话）；3. 适当增加 Epochs 到 5；4. 调整系统提示词。

**Q：为什么模型对话时回复很短/很长？**
> 调整「最大生成长度」参数；同时检查训练数据中的消息长度分布是否合理。

**Q：能用训练好的模型做更多事吗？**
> LoRA adapter 可导入 LM Studio、Ollama 等工具，也可以通过 llamafactory 导出合并后的完整模型。

**Q：界面提示检测到 NVIDIA 显卡，但 PyTorch 未启用 CUDA？**
> 说明显卡驱动或系统已能识别 GPU（EchoSelf 通过 `nvidia-smi` 可见），但当前 Python 环境里的 PyTorch 是 **CPU 构建** 或未正确链接 CUDA。请按「📖 帮助文档」中 **CUDA** 小节：用 PyTorch 官网生成的 **CUDA 版** `pip` 命令重装 `torch`，验证 `torch.cuda.is_available()` 为 `True` 后重启本程序。

---

### 💻 硬件参考

| 设备 | 可训练模型（约） | 备注 |
|------|------------------|------|
| Apple Silicon 统一内存 ~16GB | 0.5B ~ 3B | 以本页「预设模型」动态标注为准 |
| Apple Silicon 统一内存 ~24GB+ | 约至 7B | 视实际占用与批次设置 |
| NVIDIA 10GB 级显存 | 约至 7B（QLoRA 等） | |
| NVIDIA 24GB 级显存 | 约至 14B | |
| 数据中心大显存 | 更大规模 | |

</div>
""", elem_classes=["help-content"])

        # ── 切换到训练/对话 Tab 时自动刷新模型列表 ──────────────
        def _auto_refresh_train():
            """切换到训练 Tab 时自动扫描本地模型并更新下拉列表"""
            models = scan_local_models()
            hint = (
                "" if models
                else "⚠️ `./models/` 目录中暂无模型，请先在「⬇️ 模型下载」Tab 下载模型。"
            )
            return gr.update(choices=models, value=models[0] if models else None), hint

        def _auto_refresh_inf():
            """切换到对话 Tab 时自动刷新模型和 adapter 列表"""
            return (
                gr.update(choices=scan_local_models()),
                gr.update(choices=scan_local_adapters()),
            )

        tab_train.select(_auto_refresh_train, outputs=[model_choice, no_model_hint])
        tab_inf.select(_auto_refresh_inf, outputs=[inf_base_model, inf_adapter])

        # ── 全局系统提示词双向同步 ────────────────────────────────
        # 全局修改 → 同步到各 Tab
        _sync_targets = [system_prompt_input, train_system_input, inf_system, ollama_system]

        def _sync_global(val):
            """将全局系统提示词同步到所有子 Tab"""
            return (val,) * len(_sync_targets)

        global_system_prompt.change(
            _sync_global,
            inputs=[global_system_prompt],
            outputs=_sync_targets,
        )

        # 任意 Tab 修改 → 反向同步回全局（方便局部调整后保持一致）
        def _sync_back(val):
            return val

        for _comp in _sync_targets:
            _comp.change(_sync_back, inputs=[_comp], outputs=[global_system_prompt])

    return demo


def main():
    import os
    import socket

    def _pick_listen_port(start: int, span: int = 80) -> tuple[int, bool]:
        """从 start 起尝试绑定 0.0.0.0，返回 (端口, 是否偏离了起始端口)。"""
        for p in range(start, start + span):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", p))
            except OSError:
                continue
            return p, p != start
        raise OSError(
            f"在 {start}–{start + span - 1} 范围内无可用监听端口。"
            "请关闭占用进程，或设置环境变量 GRADIO_SERVER_PORT 指定其它起始端口。"
        )

    preferred = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    server_port, bumped = _pick_listen_port(preferred, 80)
    if bumped:
        print(f"提示：端口 {preferred} 已被占用，已改用 {server_port}。")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
