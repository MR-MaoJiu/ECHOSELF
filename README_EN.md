# EchoSelf

Extract conversation data from chat logs, fine-tune a language model, and build your own AI digital twin.

> **⚠️ Important Notice**: This project is intended for **educational and research purposes only**. Before using it, ensure you have obtained explicit consent from all parties involved in the chat data, and comply with all applicable laws and regulations in your jurisdiction. Do not use this project for any illegal, infringing, or privacy-violating activities.

---

## Overview

EchoSelf is a fully local, GUI-based tool for processing chat history and training language models. Simply export your chat records as JSON files, drop them into a folder, and use the GUI to clean, filter, anonymize, and convert them into a training dataset compatible with LLaMA-Factory — with built-in model downloading and one-click training.

**Everything runs locally. Your data never leaves your machine.**

---

## Project Structure

```
echoself/
├── app.py                  # GUI entry point (Gradio)
├── pyproject.toml          # Dependencies and project metadata
├── README.md
├── models/                 # Local model storage (created at runtime)
├── src/
│   ├── __init__.py
│   ├── parser.py           # Chat log parsing layer
│   ├── preprocessor.py     # Data processing layer (filter / anonymize / QA pairing)
│   └── trainer.py          # Training wrapper (LLaMA-Factory interface + model download)
└── output/                 # Created at runtime
    ├── sft_data.json       # Processed training data
    ├── dataset_info.json   # LLaMA-Factory dataset registration file
    ├── train_args_snapshot.json  # Training config snapshot (for reproducibility)
    └── model/              # Fine-tuned model weights (LoRA adapter)
```

---

## Input Data Format

EchoSelf supports the WeChat chat export format from **[WeFlow](https://github.com/re-collect-cn/weflow)**. Each contact corresponds to one `.json` file with the following structure:

```json
{
  "session": {
    "wxid": "wxid_xxxxxxxxxx",
    "displayName": "Contact Name",
    "nickname": "Contact Nickname (fallback)",
    "type": "Private Chat"
  },
  "messages": [
    {
      "localId": 1,
      "createTime": 1700000000,
      "formattedTime": "2023-11-15 10:00:00",
      "type": "Text",
      "content": "Message content here",
      "isSend": 1,
      "senderUsername": "wxid_xxxxxxxxxx",
      "senderDisplayName": "Your Name",
      "quotedContent": null
    }
  ]
}
```

**Field Reference:**

| Field | Type | Description |
|-------|------|-------------|
| `session.wxid` | string | Contact's WeChat account ID |
| `session.displayName` | string | Contact's display name |
| `session.type` | string | Chat type: `Private Chat` / `Group Chat` |
| `messages[].isSend` | int | `1` = sent by you, `0` = received |
| `messages[].type` | string | Supported: `Text`, `Quote`; others (image, voice, etc.) are filtered out |
| `messages[].content` | string | Message text content |
| `messages[].senderUsername` | string | Sender's account ID (used to auto-detect your own ID) |
| `messages[].quotedContent` | string\|null | Quoted message content (for reply messages only) |

> **How to export?** Use [WeFlow](https://github.com/re-collect-cn/weflow) or similar tools to export WeChat chat history. Each contact is exported as a separate `.json` file. Put all files into a single folder and import it into EchoSelf.

---

## GUI Tabs

### 📦 Data Processing

- **Native folder picker**: Click "📂 Select Folder" to open a system dialog — no manual path typing needed
- **Auto account ID detection**: After selecting a folder, EchoSelf scans the chat records and automatically detects your WeChat ID from sent messages
- Configurable parameters: time window, message length filter, blocked words, system prompt, and more
- Calls `parser` + `preprocessor` to process data and displays statistics and a preview
- Download the generated training data file (`sft_data.json`)

### ⬇️ Model Download

- Supports **ModelScope** (recommended for users in China, no proxy required) and **HuggingFace**
- Select a preset model and the model ID and local save path are filled automatically
- Streaming download logs with stop support
- Preset model list with M4 memory requirements:

| Model | Memory | Mac M4 16GB |
|-------|--------|-------------|
| Qwen2.5-0.5B-Instruct | ~2 GB | ✅ |
| Qwen2.5-1.5B-Instruct | ~4 GB | ✅ Recommended |
| Qwen2.5-3B-Instruct | ~8 GB | ✅ |
| Qwen2.5-7B-Instruct | ~16 GB | ⚠️ Tight |
| Qwen2.5-14B-Instruct | ~32 GB | ❌ |
| Llama-3.2-1B-Instruct | ~3 GB | ✅ |
| Llama-3.2-3B-Instruct | ~8 GB | ✅ |
| SmolLM2-1.7B-Instruct | ~4 GB | ✅ |
| Phi-3.5-mini-instruct | ~8 GB | ✅ |

### 🎯 Model Training

- Detects local LLaMA-Factory installation
- Training hyperparameter form: model path, LoRA rank, learning rate, epochs, etc.
- **Apple Silicon auto-adaptation**: Automatically switches to `bf16` precision, disables Flash Attention, and uses MPS backend on M-series chips
- Preview the full training command before running
- One-click training with real-time log streaming and stop support
- **Live loss curve** plot and training health indicator (🟢 / 🟡 / 🔴)

### 💬 Model Chat

- Load a base model + optional LoRA adapter for interactive inference
- Supports streaming token-by-token responses
- Configurable: system prompt, temperature, max new tokens
- One-click unload to free GPU/MPS memory

### 📖 Help Docs

- In-app documentation explaining the full workflow, technical terms (LoRA, SFT, bf16, etc.), training quality evaluation, Apple Silicon recommendations, FAQ, and hardware reference

---

## Module Reference

### `src/parser.py` — Chat Log Parser

**Responsibility:** Parse raw JSON chat files into structured Python objects, abstracting away raw data format details.

**Core data structures:**

```
ChatMessage
  ├── local_id       Message sequence number
  ├── create_time    Unix timestamp (seconds)
  ├── formatted_time Human-readable time string
  ├── msg_type       Message type (Text / Image / ...)
  ├── content        Text content
  ├── is_send        Direction (1=sent by you, 0=received)
  ├── sender_id      Sender account ID
  ├── sender_name    Sender display name
  └── quoted_content Quoted message content (reply messages only)

ChatSession
  ├── my_id          Your account ID (auto-detected or manually specified)
  ├── contact_name   Contact display name
  ├── contact_id     Contact account ID
  ├── chat_type      Chat type (Private / Group)
  └── messages       List of ChatMessage
```

**Key functions:**

| Function | Description |
|----------|-------------|
| `parse_json_file(path, my_id)` | Parse a single JSON file, returns `ChatSession` |
| `parse_multiple_files(paths, my_id)` | Batch parse, returns list of `ChatSession` |
| `get_message_type_stats(session)` | Count message type distribution (for GUI preview) |

---

### `src/preprocessor.py` — Data Processing Layer

**Responsibility:** Run a five-stage processing pipeline on parsed messages to produce LLaMA-Factory-compatible training data.

**Pipeline:**

```
Raw messages
    │
    ▼ 1. Type filter
    │  Keep: Text, Quote messages
    │  Drop: Images, videos, voice, files, system messages, etc.
    │
    ▼ 2. Length filter + placeholder filter
    │  Drop messages that are too short/long or purely placeholder text ("[Image]", "[Voice]", etc.)
    │
    ▼ 3. PII anonymization
    │  Regex replacement: phone numbers, emails, ID cards, bank cards, QQ IDs, IP addresses, etc.
    │  Blocked word filter: messages containing blocked words are removed entirely
    │
    ▼ 4. Consecutive message merging
    │  Messages from the same sender within a time window are concatenated into one
    │
    ▼ 5. QA pair matching
       Within a time window: contact message (instruction) + your reply (output) → QAPair
```

**PII Anonymization Rules:**

| Type | Replacement |
|------|-------------|
| Phone number (11-digit) | `[PHONE]` |
| Landline number | `[TEL]` |
| Email address | `[EMAIL]` |
| National ID (18-digit) | `[ID_CARD]` |
| Bank card number | `[BANK_CARD]` |
| QQ number | `[QQ]` |
| Account ID | `[ACCOUNT]` |
| IP address | `[IP]` |
| Home address | `[ADDRESS]` |

**Output formats:**

*Alpaca format (default):*

```json
{
  "instruction": "What the contact said",
  "input": "",
  "output": "Your reply",
  "system": "You are a real person named..."
}
```

*ShareGPT format:*

```json
{
  "conversations": [
    {"from": "human", "value": "What the contact said"},
    {"from": "gpt",   "value": "Your reply"}
  ],
  "system": "You are a real person named..."
}
```

---

### `src/trainer.py` — Training Wrapper Layer

**Responsibility:** Wrap LLaMA-Factory CLI calls, manage training subprocesses; also wrap model download subprocesses supporting ModelScope / HuggingFace.

**Core class: `TrainingProcess`**

```
TrainingProcess
  ├── start(config, log_callback, done_callback)
  │     Detect env → write config → start subprocess → stream stdout in background thread
  ├── stop()
  │     Send SIGTERM to terminate
  └── is_running (property)
        Whether the process is still alive
```

**Core class: `DownloadProcess`**

```
DownloadProcess
  ├── start(source, model_id, local_dir, log_callback, done_callback)
  │     source="modelscope" → calls modelscope download CLI
  │     source="huggingface" → calls huggingface-cli / huggingface_hub
  ├── stop()
  └── is_running (property)
```

**Config class: `TrainConfig`**

| Group | Key Fields |
|-------|-----------|
| Model | `model_name_or_path`, `template` |
| Data | `dataset_path`, `dataset_dir`, `output_dir` |
| LoRA | `lora_rank`, `lora_dropout`, `lora_target` |
| Hyperparams | `num_train_epochs`, `learning_rate`, `batch_size`, `cutoff_len` |
| Precision | `fp16` / `bf16` (auto-selected by device), `flash_attn` |

---

## Quick Start

### Step 1 — Create a virtual environment

```sh
cd echoself
uv venv .venv --python=3.12
source .venv/bin/activate
```

### Step 2 — Install dependencies

**Users in China (Tsinghua mirror, no proxy needed):**

```sh
uv pip install -e ".[all]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**International users:**

```sh
uv pip install -e ".[all]"
```

> `[all]` installs: `gradio` · `pandas` · `modelscope` · `huggingface-hub` · `llamafactory` (which includes `torch` / `transformers` / `peft` / `trl` / `accelerate` / `datasets` and all other ML dependencies)

### Install only what you need

| Use case | Command |
|----------|---------|
| Data processing only | `uv pip install -e "."` |
| Data processing + ModelScope downloads | `uv pip install -e ".[modelscope]"` |
| Data processing + HuggingFace downloads | `uv pip install -e ".[huggingface]"` |
| Full (including training) | `uv pip install -e ".[all]"` |

### Step 3 — Launch the GUI

```sh
.venv/bin/python app.py
# Open http://localhost:7861 in your browser
```

---

## Full Workflow

1. **Export chat records** — Use WeFlow or a similar tool to export WeChat history as `.json` files, all in one folder
2. **Process data** — In the "📦 Data Processing" tab, click "📂 Select Folder"; your account ID is auto-detected; configure parameters and click "Start Processing"
3. **Review data** — Check statistics and preview to verify data quality; download `output/sft_data.json` if needed
4. **Download a base model** — In the "⬇️ Model Download" tab, pick a model and source, click "Start Download"; saved to `./models/`
5. **Start training** — In the "🎯 Model Training" tab, configure hyperparameters and click "Start Training"

---

## Apple Silicon Notes

On M-series chips, EchoSelf automatically handles the following — no manual configuration required:

| Item | What happens automatically |
|------|---------------------------|
| Training backend | PyTorch MPS (Metal Performance Shaders) |
| Precision | Switched to `bf16` (MPS does not support `fp16`) |
| Flash Attention | Disabled (incompatible with MPS) |
| Recommended models | Qwen2.5-1.5B / 3B (for M4 16GB) |

**M4 16GB memory reference:**

| Model | LoRA Training Memory | Recommended? |
|-------|---------------------|--------------|
| Qwen2.5-0.5B | ~2 GB | ✅ |
| Qwen2.5-1.5B | ~4 GB | ✅ Best choice |
| Qwen2.5-3B | ~8 GB | ✅ |
| Qwen2.5-7B | ~16 GB | ⚠️ Fills RAM |

---

## GPU Reference (Non-Apple)

| Method | 7B Model | 14B Model |
|--------|----------|-----------|
| LoRA (fp16) | ~16 GB VRAM | ~32 GB VRAM |
| QLoRA (4-bit) | ~6 GB VRAM | ~12 GB VRAM |

---

## Dependencies

### Direct dependencies (`pyproject.toml`)

| Package | Version | Purpose | Group |
|---------|---------|---------|-------|
| `gradio` | ≥ 5.0.0 | GUI framework | Required |
| `pandas` | ≥ 2.0.0 | Data processing | Required |
| `modelscope` | ≥ 1.9.0 | Model download (China) | `[modelscope]` |
| `huggingface-hub` | ≥ 0.20.0 | Model download (international) | `[huggingface]` |
| `llamafactory` | ≥ 0.9.0 | Fine-tuning engine | `[train]` |

### ML dependencies installed by `llamafactory`

| Package | Description |
|---------|-------------|
| `torch` | PyTorch, includes MPS support for Apple Silicon |
| `transformers` | HuggingFace model loading and inference |
| `peft` | LoRA / QLoRA parameter-efficient fine-tuning |
| `trl` | SFT Trainer and RLHF utilities |
| `accelerate` | Multi-device training acceleration |
| `datasets` | Dataset loading and processing |
| `tokenizers` | Fast tokenizer implementation |
| `safetensors` | Safe model weight format |
| `sentencepiece` | SentencePiece tokenizer (required by some models) |

---

## ☕ Support the Project

Built this out of pure curiosity and a hope it might be useful to someone. A GitHub star already means a lot ⭐

If you'd like to go further — there's a code below, but absolutely no pressure 😄

<div align="center">
  <img src="./assets/donate.png" alt="Donate" width="260" />
</div>

---

## ⚖️ Legal Disclaimer

1. **Educational use only**: This project is an open-source learning tool. Commercial use, illegal activity, or any action that infringes on others' rights is strictly prohibited.

2. **Data compliance**: Before processing any chat history, ensure that:
   - You have obtained **explicit consent** from all parties involved
   - You comply with applicable data protection laws (e.g., GDPR, PIPL, CCPA)
   - You do not process, train on, or distribute private data involving minors

3. **Privacy protection**: Built-in PII anonymization is available — enable it. Keep your training data and model weights secure and do not share them publicly.

4. **Disclaimer**: The authors are not liable for any direct or indirect damages arising from use of this project. Users assume full legal responsibility.

---

## 📄 License

This project is licensed under the **MIT License with Attribution Requirement**.

**You are free to:**
- ✅ Use, copy, modify, and distribute this code
- ✅ Build derivative works and open-source them
- ✅ Use modified versions for personal learning and research

**You must:**
- 📌 Prominently credit the original project in your documentation, README, or UI:
  ```
  Based on EchoSelf (https://github.com/MR-MaoJiu/ECHOSELF)
  ```
- 📌 Retain the original copyright notice and this license text

See [LICENSE](./LICENSE) for full details.
