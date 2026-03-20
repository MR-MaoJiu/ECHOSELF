# EchoSelf

从聊天记录提取对话数据，微调语言模型，打造专属数字分身。

---

## 项目概览

EchoSelf 是一个本地化、可视化的聊天记录数据处理与模型训练工具。只需将导出的聊天记录 JSON 文件放入指定文件夹，通过 GUI 完成数据清洗、格式转换、隐私脱敏，最终生成可直接用于 LLaMA-Factory 微调的训练数据集，并支持一键下载模型、启动训练。

全程本地运行，数据不离开本机。

---

## 完整架构

```
echoself/
├── app.py                  # GUI 入口（Gradio）
├── pyproject.toml          # 项目依赖与元信息
├── README.md
├── models/                 # 本地模型存放目录（运行时生成）
├── src/
│   ├── __init__.py
│   ├── parser.py           # 聊天记录解析层
│   ├── preprocessor.py     # 数据处理层（过滤 / 脱敏 / QA 构建）
│   └── trainer.py          # 训练封装层（LLaMA-Factory 接口 + 模型下载）
└── output/                 # 运行时生成
    ├── sft_data.json       # 处理好的训练数据
    ├── dataset_info.json   # LLaMA-Factory 数据集注册文件
    ├── train_args_snapshot.json  # 训练参数快照（方便复现）
    └── model/              # 微调后模型权重（LoRA adapter）
```

---

## GUI 功能页

### 📦 数据处理

- **原生文件夹选择**：点击「📂 选择文件夹」弹出系统对话框，无需手动输入路径
- **账号 ID 自动识别**：选择文件夹后立即扫描聊天记录，从发出的消息中提取微信账号 ID 并自动填入
- 提供参数配置表单（时间窗口、长度过滤、禁用词、系统提示词等）
- 调用 `parser` + `preprocessor` 完成数据处理，展示统计结果和数据预览
- 支持下载生成的训练数据文件

### ⬇️ 模型下载

- **ModelScope**（推荐国内用户，无需代理）和 **HuggingFace** 双源可切换
- 从预设模型列表选择后自动填写模型 ID 和本地保存路径
- 流式展示下载日志，支持中途停止
- 预设模型（均标注 M4 内存需求）：

| 模型 | 内存占用 | Mac M4 16GB |
|------|---------|------------|
| Qwen2.5-0.5B-Instruct | ~2 GB | ✅ |
| Qwen2.5-1.5B-Instruct | ~4 GB | ✅ 推荐 |
| Qwen2.5-3B-Instruct | ~8 GB | ✅ |
| Qwen2.5-7B-Instruct | ~16 GB | ⚠️ 需独显 |
| Qwen2.5-14B-Instruct | ~32 GB | ❌ |
| Llama-3.2-1B-Instruct | ~3 GB | ✅ |
| Llama-3.2-3B-Instruct | ~8 GB | ✅ |
| SmolLM2-1.7B-Instruct | ~4 GB | ✅ |
| Phi-3.5-mini-instruct | ~8 GB | ✅ |

### 🎯 模型训练

- 检测本机 LLaMA-Factory 安装状态
- 提供训练超参配置表单（模型路径、LoRA rank、学习率、Epochs 等）
- **Apple Silicon 自动适配**：检测到 M 系列芯片时自动切换为 bf16 精度、关闭 Flash Attention、使用 MPS 后端
- 预览完整训练命令
- 一键启动训练子进程，流式展示训练日志，支持中途停止

---

## 模块说明

### `src/parser.py` — 聊天记录解析层

**职责：** 将原始 JSON 聊天记录文件解析为结构化对象，屏蔽底层数据格式细节。

**核心数据结构：**

```
ChatMessage
  ├── local_id       消息序号
  ├── create_time    Unix 时间戳（秒）
  ├── formatted_time 格式化时间字符串
  ├── msg_type       消息类型（文本消息 / 图片消息 / ...）
  ├── content        消息文本内容
  ├── is_send        方向（1=自己发，0=对方发）
  ├── sender_id      发送方账号 ID
  ├── sender_name    发送方昵称
  └── quoted_content 被引用的原文（引用消息专用）

ChatSession
  ├── my_id          自己的账号 ID（自动推断或手动指定）
  ├── contact_name   对方昵称
  ├── contact_id     对方账号 ID
  ├── chat_type      对话类型（私聊 / 群聊）
  └── messages       ChatMessage 列表
```

**主要函数：**

| 函数 | 说明 |
|------|------|
| `parse_json_file(path, my_id)` | 解析单个 JSON 文件，返回 `ChatSession` |
| `parse_multiple_files(paths, my_id)` | 批量解析，返回 `ChatSession` 列表 |
| `get_message_type_stats(session)` | 统计消息类型分布（用于 GUI 预览） |

**支持的 JSON 格式：** WeFlow 导出的私聊格式（`messages` 数组 + `session` 元信息）

---

### `src/preprocessor.py` — 数据处理层

**职责：** 对解析后的消息执行五步处理流程，最终输出 LLaMA-Factory 兼容的训练数据。

**处理流程：**

```
原始消息列表
    │
    ▼ 1. 类型过滤
    │  保留：文本消息、引用消息
    │  丢弃：图片、视频、语音、文件、系统消息 等
    │
    ▼ 2. 长度过滤 + 占位内容过滤
    │  丢弃过短/过长消息，丢弃 "[图片]"、"[语音]" 等纯占位内容
    │
    ▼ 3. 隐私脱敏（PII Removal）
    │  正则替换：手机号、邮箱、身份证、银行卡、QQ 号、IP 地址等
    │  禁用词过滤：命中整条消息删除
    │
    ▼ 4. 连续消息合并
    │  同一发送方在时间窗口内的多条消息拼接为一条
    │
    ▼ 5. QA 对匹配
       时间窗口内：对方消息(instruction) + 自己回复(output) → QAPair
```

**隐私脱敏规则：**

| 类型 | 替换占位符 |
|------|-----------|
| 手机号（11位） | `[手机号]` |
| 座机号 | `[座机号]` |
| 电子邮箱 | `[邮箱]` |
| 身份证号（18位） | `[身份证]` |
| 银行卡号 | `[银行卡]` |
| QQ 号 | `[QQ号]` |
| 账号 ID | `[账号]` |
| IP 地址 | `[IP地址]` |
| 家庭地址 | `[地址]` |

**输出格式：**

*Alpaca 格式（默认）：*

```json
{
  "instruction": "对方说的话",
  "input": "",
  "output": "我的回复",
  "system": "请你扮演一个真实的人..."
}
```

*ShareGPT 格式：*

```json
{
  "conversations": [
    {"from": "human", "value": "对方说的话"},
    {"from": "gpt",   "value": "我的回复"}
  ],
  "system": "请你扮演一个真实的人..."
}
```

**主要函数：**

| 函数 | 说明 |
|------|------|
| `build_qa_pairs(session, ...)` | 完整处理流程，返回 `ProcessResult`（含统计信息和 QA 对列表） |
| `remove_pii(text, blocked_words)` | 单条文本脱敏 |
| `save_dataset(qa_pairs, path, fmt)` | 序列化保存 JSON 训练数据 |

---

### `src/trainer.py` — 训练封装层

**职责：** 封装 LLaMA-Factory CLI 调用，管理训练子进程；同时封装模型下载子进程，支持 ModelScope / HuggingFace 双源。

**核心类：`TrainingProcess`**

```
TrainingProcess
  ├── start(config, log_callback, done_callback)
  │     检测环境 → 写配置文件 → 启动子进程 → 后台线程流式读取 stdout
  ├── stop()
  │     发送 SIGTERM 终止进程
  └── is_running (property)
        返回进程是否仍在运行
```

**核心类：`DownloadProcess`**

```
DownloadProcess
  ├── start(source, model_id, local_dir, log_callback, done_callback)
  │     source="modelscope" → 调用 modelscope download CLI
  │     source="huggingface" → 调用 huggingface-cli / huggingface_hub
  ├── stop()
  │     发送 SIGTERM 终止进程
  └── is_running (property)
```

**配置类：`TrainConfig`**

| 参数组 | 主要字段 |
|--------|---------|
| 模型 | `model_name_or_path`, `template` |
| 数据 | `dataset_path`, `dataset_dir`, `output_dir` |
| LoRA | `lora_rank`, `lora_dropout`, `lora_target` |
| 训练超参 | `num_train_epochs`, `learning_rate`, `batch_size`, `cutoff_len` |
| 精度 | `fp16` / `bf16`（设备自动适配）, `flash_attn` |

**模型下载 ID 映射：`MODEL_DOWNLOAD_IDS`**

预设了 9 款模型在 ModelScope 和 HuggingFace 上的完整 ID，切换下载源时自动替换。

**辅助函数：**

| 函数 | 说明 |
|------|------|
| `check_llamafactory()` | 检测 LLaMA-Factory 是否已安装 |
| `check_modelscope()` | 检测 modelscope 是否已安装 |
| `check_huggingface_hub()` | 检测 huggingface-hub 是否已安装 |
| `get_device_info()` | 检测当前设备（Apple Silicon / CUDA / CPU）并返回适配参数 |
| `get_train_command(config)` | 生成等价的 bash 训练命令 |
| `prepare_train_files(config)` | 写入 `dataset_info.json` 和参数快照 |

---

## 快速开始

### 第一步：创建虚拟环境

```sh
cd echoself
uv venv .venv --python=3.12
source .venv/bin/activate
```

### 第二步：一键安装全部依赖

**国内用户（清华镜像，无需代理，推荐）：**

```sh
uv pip install -e ".[all]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**境外用户：**

```sh
uv pip install -e ".[all]"
```

> `[all]` 会自动安装：`gradio` · `pandas` · `modelscope` · `huggingface-hub` · `llamafactory`（含 `torch` / `transformers` / `peft` / `trl` / `accelerate` / `datasets` 等全部 ML 依赖）

### 按需安装（轻量启动）

| 场景 | 命令 |
|------|------|
| 仅数据处理 | `uv pip install -e "."` |
| 数据处理 + 国内下载模型 | `uv pip install -e ".[modelscope]"` |
| 数据处理 + 境外下载模型 | `uv pip install -e ".[huggingface]"` |
| 全功能（含训练） | `uv pip install -e ".[all]"` |

### 第三步：启动 GUI

```sh
.venv/bin/python app.py
# 浏览器访问 http://localhost:7861
```

---

## 完整使用流程

1. **导出聊天记录** — 用 WeFlow 等工具导出 `.json` 格式聊天记录，放在同一个文件夹
2. **数据处理** — 在「📦 数据处理」Tab 点击「📂 选择文件夹」，账号 ID 自动识别，配置参数后点击「开始处理」
3. **检查数据** — 查看统计信息和预览，确认数据质量，可下载 `output/sft_data.json`
4. **下载基础模型** — 在「⬇️ 模型下载」Tab 选择模型和下载源，点击「开始下载」，模型保存至 `./models/`
5. **启动训练** — 在「🎯 模型训练」Tab 配置参数，点击「开始训练」

---

## Apple Silicon 适配说明

在 M 系列芯片上运行时，EchoSelf 会自动完成以下适配，无需手动设置：

| 项目 | 自动处理 |
|------|---------|
| 训练后端 | PyTorch MPS（Metal Performance Shaders） |
| 精度 | 自动切换为 bf16（MPS 不支持 fp16） |
| Flash Attention | 自动禁用（MPS 不兼容） |
| 推荐模型 | Qwen2.5-1.5B / 3B（M4 16GB 推荐） |

**M4 16GB 内存参考：**

| 模型 | LoRA 训练内存 | 是否推荐 |
|------|------------|---------|
| Qwen2.5-0.5B | ~2 GB | ✅ |
| Qwen2.5-1.5B | ~4 GB | ✅ 首选 |
| Qwen2.5-3B | ~8 GB | ✅ |
| Qwen2.5-7B | ~16 GB | ⚠️ 内存占满 |

---

## 硬件参考（GPU）

| 方法 | 7B 模型 | 14B 模型 |
|------|---------|---------|
| LoRA（fp16） | ~16 GB 显存 | ~32 GB 显存 |
| QLoRA（4-bit） | ~6 GB 显存 | ~12 GB 显存 |

---

## 依赖说明

### 直接依赖（`pyproject.toml` 中声明）

| 包 | 版本要求 | 用途 | 分组 |
|----|---------|------|------|
| `gradio` | ≥ 5.0.0 | GUI 框架 | 必须 |
| `pandas` | ≥ 2.0.0 | 数据处理 | 必须 |
| `modelscope` | ≥ 1.9.0 | 国内模型下载 | `[modelscope]` |
| `huggingface-hub` | ≥ 0.20.0 | 境外模型下载 | `[huggingface]` |
| `llamafactory` | ≥ 0.9.0 | 模型微调训练引擎 | `[train]` |

### 由 `llamafactory` 自动安装的 ML 依赖

| 包 | 说明 |
|----|------|
| `torch` | PyTorch，含 MPS（Apple Silicon）支持 |
| `transformers` | HuggingFace 模型加载与推理 |
| `peft` | LoRA / QLoRA 参数高效微调 |
| `trl` | 强化学习微调（SFT Trainer） |
| `accelerate` | 多设备训练加速 |
| `datasets` | 数据集加载与处理 |
| `tokenizers` | 高速分词器 |
| `safetensors` | 模型权重安全格式 |
| `sentencepiece` | SentencePiece 分词（部分模型需要） |
