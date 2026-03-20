"""
EchoSelf — 从聊天记录训练数字分身
Gradio GUI 主程序
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from src.parser import get_message_type_stats, parse_multiple_files
from src.preprocessor import build_qa_pairs, save_dataset
from src.trainer import (
    MODEL_DOWNLOAD_IDS,
    MODEL_PRESETS,
    DownloadProcess,
    TrainConfig,
    TrainingProcess,
    check_huggingface_hub,
    check_llamafactory,
    check_modelscope,
    get_device_info,
    get_template_for_model,
    get_train_command,
)

OUTPUT_DIR = Path("./output")
DATASET_PATH = OUTPUT_DIR / "sft_data.json"

_training_process = TrainingProcess()
_train_logs: list[str] = []

_download_process = DownloadProcess()
_download_logs: list[str] = []

# 启动时检测一次设备及下载工具
_DEVICE_INFO = get_device_info()
_MS_OK  = check_modelscope()
_HF_OK  = check_huggingface_hub()


# ──────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────

def _open_native_folder_picker() -> str:
    """
    用 osascript 弹出 macOS 原生文件夹选择框。
    在子进程中执行，不阻塞 Gradio 工作线程，避免连接超时。
    用户取消返回空字符串。
    """
    import subprocess
    script = 'POSIX path of (choose folder with prompt "选择聊天记录文件夹")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,  # 最长等待 5 分钟
        )
        if result.returncode == 0:
            # osascript 返回结果末尾带 '\n'，路径末尾带 '/'，均去掉
            return result.stdout.strip().rstrip("/")
        return ""
    except Exception:
        return ""


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
    folder_str = _open_native_folder_picker()
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
    )


def start_training(
    model_name, model_path, template, dataset_path, output_dir,
    system_prompt, lora_rank, epochs, batch_size, grad_accum,
    learning_rate, cutoff_len,
):
    global _train_logs
    _train_logs = []

    if not Path(dataset_path).exists():
        yield "❌ 训练数据文件不存在，请先完成数据处理步骤。"
        return

    cfg = _make_config(
        model_name, model_path, template, dataset_path, output_dir,
        system_prompt, lora_rank, epochs, batch_size, grad_accum,
        learning_rate, cutoff_len,
    )

    done_flag: dict = {"done": False, "code": -1}

    def on_log(line: str):
        _train_logs.append(line)

    def on_done(code: int):
        done_flag["done"] = True
        done_flag["code"] = code

    started = _training_process.start(cfg, on_log, on_done)
    if not started:
        yield "\n".join(_train_logs)
        return

    import time
    while not done_flag["done"]:
        time.sleep(1)
        yield "\n".join(_train_logs[-100:])

    suffix = "✅ 训练完成！" if done_flag["code"] == 0 else f"❌ 训练异常退出（code={done_flag['code']}）"
    _train_logs.append(f"\n{suffix}  模型保存在：{output_dir}")
    yield "\n".join(_train_logs[-100:])


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

footer { display: none !important; }
"""

# 模型选项：显示文本 = "名称  (内存需求)"
_MODEL_CHOICES = [f"{name}  ({note})" for name, _, note in MODEL_PRESETS]
_MODEL_NAMES   = [name for name, _, _ in MODEL_PRESETS]
_DEFAULT_MODEL = _DEVICE_INFO["default_model"]
_DEFAULT_CHOICE = next(
    (c for c in _MODEL_CHOICES if _DEFAULT_MODEL in c),
    _MODEL_CHOICES[1],  # fallback: 1.5B
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EchoSelf") as demo:

        gr.Markdown("# 🪞 EchoSelf\n**从聊天记录训练数字分身。**")

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
                            label="系统提示词（训练时注入）",
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
                        info="标注 ✅ 的模型可在 M4 16GB 上正常训练",
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
            with gr.Tab("🎯 模型训练"):

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
                        info="显示 ./models/ 下已下载的模型，点击「🔄 刷新」重新扫描",
                        allow_custom_value=True,
                        scale=3,
                    )
                    refresh_model_btn = gr.Button("🔄 刷新", variant="secondary", scale=1, min_width=80)
                    model_path_input = gr.Textbox(
                        label="自定义模型路径（留空则使用上方选中路径）",
                        placeholder="/path/to/your/model",
                        scale=3,
                    )
                    template_input = gr.Textbox(
                        label="对话模板（自动填写）",
                        value=get_template_for_model(_local_default or ""),
                        scale=1,
                    )

                no_model_hint = gr.Markdown(_no_model_hint)

                gr.Markdown("### 数据 & 输出")
                with gr.Row():
                    dataset_path_input = gr.Textbox(
                        label="训练数据路径",
                        value=str(DATASET_PATH),
                        scale=3,
                    )
                    output_dir_input = gr.Textbox(
                        label="模型输出目录",
                        value="./output/model",
                        scale=2,
                    )

                train_system_input = gr.Textbox(
                    label="系统提示词",
                    value="请你扮演一个真实的人，用自然的方式进行对话。",
                    lines=2,
                )

                gr.Markdown("### 训练超参")
                with gr.Row():
                    lora_rank_input    = gr.Slider(4, 64, value=8,   step=4,   label="LoRA Rank")
                    epochs_input       = gr.Slider(1, 10, value=3,   step=0.5, label="训练轮数")
                    batch_size_input   = gr.Slider(1, 8,  value=1,   step=1,   label="Batch Size（Mac 建议 1）")
                    grad_accum_input   = gr.Slider(1, 32, value=8,   step=1,   label="梯度累积步数")
                with gr.Row():
                    lr_input           = gr.Number(value=1e-4, label="学习率")
                    cutoff_input       = gr.Slider(128, 2048, value=512, step=128, label="最大序列长度（Mac 建议 512）")

                gr.Markdown(
                    f"> **精度设置由设备自动决定**：当前设备为 **{_DEVICE_INFO['device']}**，"
                    f"训练时将自动使用 {'bf16' if _DEVICE_INFO['use_bf16'] else 'fp16' if _DEVICE_INFO['use_fp16'] else 'fp32'}，"
                    f"Flash Attention：{_DEVICE_INFO['flash_attn']}。"
                )

                train_inputs = [
                    model_choice, model_path_input, template_input,
                    dataset_path_input, output_dir_input, train_system_input,
                    lora_rank_input, epochs_input, batch_size_input,
                    grad_accum_input, lr_input, cutoff_input,
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

                train_log_output = gr.Textbox(
                    label="训练日志",
                    lines=22,
                    max_lines=30,
                    autoscroll=True,
                    interactive=False,
                )
                stop_status = gr.Markdown()

                # model_choice 现在直接是本地路径，自定义路径优先
                def _resolve_model(choice: str, custom_path: str) -> str:
                    return custom_path.strip() if custom_path.strip() else (choice or "")

                def _wrap_start(*args):
                    resolved = _resolve_model(args[0], args[1])
                    return start_training(resolved, *args[1:])

                def _wrap_preview(*args):
                    resolved = _resolve_model(args[0], args[1])
                    return get_command_preview(resolved, *args[1:])

                cmd_preview_btn.click(_wrap_preview, inputs=train_inputs, outputs=[cmd_preview_output])
                start_btn.click(_wrap_start, inputs=train_inputs, outputs=[train_log_output])
                stop_btn.click(stop_training, outputs=[stop_status])

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
