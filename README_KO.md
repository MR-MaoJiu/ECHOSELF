<div align="right">
  <a href="./README.md">简体中文</a> ｜
  <a href="./README_EN.md">English</a> ｜
  <a href="./README_JA.md">日本語</a> ｜
  <a href="./README_KO.md">한국어</a>
</div>

# EchoSelf

채팅 기록에서 대화 데이터를 추출하고, 언어 모델을 파인튜닝하여 나만의 AI 디지털 트윈을 만듭니다.

> **⚠️ 중요 공지**: 이 프로젝트는 **학습 및 연구 목적으로만** 제공됩니다. 사용 전에 채팅 데이터에 관련된 모든 당사자의 명시적 동의를 받으셔야 합니다. 거주 지역의 법률 및 규정 범위 내에서 사용하시기 바랍니다. 불법 행위, 권리 침해 또는 개인 정보 침해 목적으로의 사용은 엄격히 금지됩니다.

---

## 개요

EchoSelf는 완전 로컬 기반의 GUI 채팅 기록 처리 및 모델 트레이닝 도구입니다. 내보낸 채팅 기록 JSON 파일을 폴더에 넣기만 하면, GUI를 통해 데이터 정제·필터링·익명화·변환을 수행하고 LLaMA-Factory에서 바로 사용할 수 있는 학습 데이터셋을 생성합니다. 모델 다운로드와 원클릭 트레이닝도 내장되어 있습니다.

**모든 작업은 로컬에서 실행됩니다. 데이터는 기기 밖으로 나가지 않습니다.**

---

## 프로젝트 구조

```
echoself/
├── app.py                  # GUI 진입점 (Gradio)
├── pyproject.toml          # 의존성 및 프로젝트 메타데이터
├── README.md
├── models/                 # 로컬 모델 저장 디렉토리 (실행 시 생성)
├── src/
│   ├── __init__.py
│   ├── parser.py           # 채팅 기록 파싱 레이어
│   ├── preprocessor.py     # 데이터 처리 레이어 (필터 / 익명화 / QA 페어 생성)
│   └── trainer.py          # 트레이닝 래퍼 (LLaMA-Factory 인터페이스 + 모델 다운로드)
└── output/                 # 실행 시 생성
    ├── sft_data.json       # 처리된 학습 데이터
    ├── dataset_info.json   # LLaMA-Factory 데이터셋 등록 파일
    ├── train_args_snapshot.json  # 트레이닝 설정 스냅샷 (재현용)
    └── model/              # 파인튜닝 완료 모델 (LoRA 어댑터)
```

---

## 입력 데이터 형식

EchoSelf는 **[WeFlow](https://github.com/re-collect-cn/weflow)**로 내보낸 WeChat 채팅 기록 JSON 형식을 지원합니다. 각 연락처는 개별 `.json` 파일에 해당하며, 구조는 다음과 같습니다:

```json
{
  "session": {
    "wxid": "wxid_xxxxxxxxxx",
    "displayName": "상대방 표시 이름",
    "nickname": "상대방 닉네임 (보조)",
    "type": "개인 채팅"
  },
  "messages": [
    {
      "localId": 1,
      "createTime": 1700000000,
      "formattedTime": "2023-11-15 10:00:00",
      "type": "텍스트 메시지",
      "content": "메시지 내용",
      "isSend": 1,
      "senderUsername": "wxid_xxxxxxxxxx",
      "senderDisplayName": "내 이름",
      "quotedContent": null
    }
  ]
}
```

**필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `session.wxid` | string | 상대방의 WeChat 계정 ID |
| `session.displayName` | string | 상대방 표시 이름 |
| `session.type` | string | 채팅 유형: `개인 채팅` / `그룹 채팅` |
| `messages[].isSend` | int | `1` = 내가 보냄, `0` = 수신 |
| `messages[].type` | string | 지원 형식: `텍스트`·`인용`; 기타(이미지·음성 등)는 필터링 |
| `messages[].content` | string | 텍스트 내용 |
| `messages[].senderUsername` | string | 발신자 계정 ID (자동 감지에 사용) |
| `messages[].quotedContent` | string\|null | 인용 메시지 원문 (인용 메시지 전용) |

> **내보내기 방법:** [WeFlow](https://github.com/re-collect-cn/weflow) 등의 도구로 WeChat 채팅 기록을 내보냅니다. 각 연락처를 개별 `.json` 파일로 저장하고, 모든 파일을 하나의 폴더에 모아 EchoSelf에 가져오세요.

---

## GUI 탭 기능

### 📦 데이터 처리

- **네이티브 폴더 선택**: "📂 폴더 선택"을 클릭하면 시스템 대화상자가 열려 경로를 직접 입력할 필요 없음
- **계정 ID 자동 감지**: 폴더 선택 후 채팅 기록을 스캔하여 전송 메시지에서 WeChat ID를 자동 감지
- 설정 매개변수: 시간 윈도우, 메시지 길이 필터, 차단 단어, 시스템 프롬프트 등
- 통계 정보 및 데이터 미리보기 표시
- 생성된 학습 데이터 파일(`sft_data.json`) 다운로드 지원

### ⬇️ 모델 다운로드

- **ModelScope**（중국 사용자 권장, 프록시 불필요）와 **HuggingFace** 이중 소스 지원
- 프리셋 모델 선택 시 모델 ID와 저장 경로 자동 입력
- 스트리밍 다운로드 로그, 중간 중지 지원
- 프리셋 모델 목록 (M4 메모리 요구량 포함):

| 모델 | 메모리 | Mac M4 16GB |
|------|--------|-------------|
| Qwen2.5-0.5B-Instruct | ~2 GB | ✅ |
| Qwen2.5-1.5B-Instruct | ~4 GB | ✅ 권장 |
| Qwen2.5-3B-Instruct | ~8 GB | ✅ |
| Qwen2.5-7B-Instruct | ~16 GB | ⚠️ 메모리 부족 가능 |
| Qwen2.5-14B-Instruct | ~32 GB | ❌ |
| Llama-3.2-1B-Instruct | ~3 GB | ✅ |
| Llama-3.2-3B-Instruct | ~8 GB | ✅ |
| SmolLM2-1.7B-Instruct | ~4 GB | ✅ |
| Phi-3.5-mini-instruct | ~8 GB | ✅ |

### 🎯 모델 트레이닝

- 로컬 LLaMA-Factory 설치 상태 자동 감지
- 트레이닝 하이퍼파라미터 폼: 모델 경로, LoRA rank, 학습률, Epoch 수 등
- **Apple Silicon 자동 최적화**: M 시리즈 칩 감지 시 `bf16` 정밀도로 자동 전환, Flash Attention 비활성화, MPS 백엔드 사용
- 실행 전 전체 트레이닝 명령 미리보기
- 원클릭 트레이닝, 실시간 로그 스트리밍, 중간 중지 지원
- **실시간 Loss 곡선** 그래프 및 트레이닝 건강 지표（🟢 / 🟡 / 🔴）

### 💬 모델 대화

- 베이스 모델 + 선택적 LoRA 어댑터를 로드하여 인터랙티브 추론
- 토큰 단위 스트리밍 응답 지원
- 설정 가능 항목: 시스템 프롬프트, 온도(temperature), 최대 생성 토큰 수
- 원클릭으로 모델 언로드하여 GPU/MPS 메모리 해제

### 📖 도움말 문서

- LoRA·SFT·bf16 등 기술 용어 해설
- 트레이닝 품질 평가 기준
- Apple Silicon 파라미터 권장값
- 자주 묻는 질문(FAQ) 및 하드웨어 참고 자료

---

## 빠른 시작

### 1단계 — 가상 환경 생성

```sh
cd echoself
uv venv .venv --python=3.12
source .venv/bin/activate
```

### 2단계 — 의존성 설치

**중국 내 사용자 (칭화 미러, 프록시 불필요):**

```sh
uv pip install -e ".[all]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**그 외 사용자:**

```sh
uv pip install -e ".[all]"
```

> `[all]`에는 다음이 포함됩니다: `gradio` · `pandas` · `modelscope` · `huggingface-hub` · `llamafactory`（`torch` / `transformers` / `peft` / `trl` / `accelerate` / `datasets` 등 ML 의존성 포함）

### 필요한 기능만 설치

| 사용 사례 | 명령어 |
|----------|--------|
| 데이터 처리만 | `uv pip install -e "."` |
| 데이터 처리 + ModelScope 다운로드 | `uv pip install -e ".[modelscope]"` |
| 데이터 처리 + HuggingFace 다운로드 | `uv pip install -e ".[huggingface]"` |
| 전체 기능 (트레이닝 포함) | `uv pip install -e ".[all]"` |

### 3단계 — GUI 실행

```sh
.venv/bin/python app.py
# 브라우저에서 http://localhost:7861 접속
```

---

## 전체 사용 흐름

1. **채팅 기록 내보내기** — WeFlow 등으로 WeChat 기록을 `.json` 형식으로 내보내고 같은 폴더에 저장
2. **데이터 처리** — "📦 데이터 처리" 탭에서 "📂 폴더 선택" 클릭. 계정 ID 자동 감지됨. 파라미터 설정 후 "처리 시작" 클릭
3. **데이터 확인** — 통계 및 미리보기로 품질 확인. 필요시 `output/sft_data.json` 다운로드
4. **베이스 모델 다운로드** — "⬇️ 모델 다운로드" 탭에서 모델과 소스를 선택하고 "다운로드 시작". `./models/`에 저장됨
5. **트레이닝 시작** — "🎯 모델 트레이닝" 탭에서 하이퍼파라미터를 설정하고 "트레이닝 시작" 클릭

---

## Apple Silicon 관련 안내

M 시리즈 칩에서 EchoSelf는 다음을 자동으로 처리합니다 — 수동 설정 불필요:

| 항목 | 자동 처리 내용 |
|------|-------------|
| 트레이닝 백엔드 | PyTorch MPS (Metal Performance Shaders) |
| 정밀도 | `bf16`으로 자동 전환 (MPS는 `fp16` 미지원) |
| Flash Attention | 자동 비활성화 (MPS와 비호환) |
| 권장 모델 | Qwen2.5-1.5B / 3B (M4 16GB 기준) |

**M4 16GB 메모리 참고:**

| 모델 | LoRA 트레이닝 메모리 | 권장 여부 |
|------|-------------------|---------|
| Qwen2.5-0.5B | ~2 GB | ✅ |
| Qwen2.5-1.5B | ~4 GB | ✅ 최적 |
| Qwen2.5-3B | ~8 GB | ✅ |
| Qwen2.5-7B | ~16 GB | ⚠️ 메모리 한계 |

---

## ☕ 후원

순수한 관심으로 만든 프로젝트입니다. 조금이나마 도움이 되었으면 좋겠습니다. GitHub 스타 하나면 충분히 힘이 납니다 ⭐

조금 더 응원해 주시고 싶다면 아래 코드가 있습니다. 스캔하지 않으셔도 전혀 괜찮습니다 😄

<div align="center">
  <img src="./assets/donate.png" alt="후원" width="260" />
</div>

---

## ⚖️ 법적 고지

1. **학습·연구 목적 전용**: 본 프로젝트는 오픈소스 학습 도구입니다. 상업적 목적, 불법 행위, 타인의 권리 침해 목적의 사용은 엄격히 금지됩니다.

2. **데이터 컴플라이언스**: 채팅 기록을 처리하기 전에 다음을 확인하세요:
   - 관련된 모든 당사자로부터 **명시적 동의**를 받았을 것
   - GDPR, 개인정보보호법 등 적용되는 법률을 준수할 것
   - 미성년자 관련 개인 데이터를 처리·학습·배포하지 않을 것

3. **개인정보 보호**: 내장된 PII 익명화 기능을 활성화하는 것을 권장합니다. 학습 데이터와 모델 가중치는 안전하게 관리하고 공개하지 마세요.

4. **면책 조항**: 개발자는 본 프로젝트 사용으로 인한 직·간접적 손해에 대해 일체의 책임을 지지 않습니다. 사용자는 사용에 따른 위험과 법적 책임을 스스로 부담합니다.

---

## 📄 라이선스

본 프로젝트는 **MIT 라이선스 (저작자 표시 요구사항 포함)** 로 오픈소스 공개됩니다.

**자유롭게 할 수 있는 것:**
- ✅ 본 프로젝트 코드 사용·복사·수정·배포
- ✅ 파생 작품을 만들어 오픈소스화
- ✅ 수정된 버전을 개인 학습·연구에 사용

**반드시 지켜야 할 것:**
- 📌 문서·README 또는 UI에 **원본 프로젝트 출처를 명시**할 것:
  ```
  Based on EchoSelf (https://github.com/MR-MaoJiu/ECHOSELF)
  ```
- 📌 원본 저작권 고지 및 본 라이선스 텍스트를 유지할 것

자세한 내용은 [LICENSE](./LICENSE) 파일을 참조하세요.
