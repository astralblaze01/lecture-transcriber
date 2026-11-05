# lecture-transcriber

로컬에서 강의 녹음을 텍스트로 바꾸고, Claude 데스크톱 앱이 그 결과로 요약하거나 **강의 노트(삼성노트·PDF)를 책처럼 정리**할 수 있게 해주는 MCP 서버예요.

> Local speech-to-text MCP server for the Claude desktop app (Windows, macOS — MLX on Apple Silicon). Transcribes lecture recordings on your own machine and lets Claude turn Samsung Notes / PDF slides + recordings into a book-style study guide.

이 도구는 음성 인식을 PC에서 실행합니다. MCP를 사용하면 Claude에는 받아쓰기 텍스트와 요청한 자료가 전달되며, 녹음 파일은 로컬에서 처리됩니다.

## 모델 직접 선택

공통 실행 옵션 `--choose-model`을 사용하면 한국어 특화 BatiSay Turbo, Whisper large-v3, Whisper large-v3-turbo 중 하나를 번호로 고를 수 있습니다. 메뉴의 기본 언어는 한국어입니다. `large-v3` 결과는 `녹음.whisper.large-v3.txt`로 따로 저장됩니다.

Claude에서는 “Whisper large-v3로 변환해줘”처럼 모델 이름을 지정하세요. 일반 녹음과 삼성 노트 모두에서 선택할 수 있습니다. `large-v3`는 `engine="whisper", model="large-v3"`로 전달됩니다.

## 기능

- **녹음 → 텍스트**: 한국어 파인튜닝 Whisper(기본), 다국어 Whisper, Cohere Transcribe 중 선택 ([변환 엔진](#변환-엔진) 참고)
- **긴 녹음 대응**: 백그라운드 작업 + 진행률 조회로 도구 호출 타임아웃 회피, 결과 캐시
- **이어하기**: 변환 중 한 줄씩 `.part`에 바로 저장해서, 중간에 꺼도 다음 실행 때 끊긴 지점부터 이어서 변환
- **삼성노트(.sdocx) 처리**: 압축을 풀고 파일 시그니처로 녹음·PDF를 골라냄
- **책 만들기**: 슬라이드 순서를 뼈대로 강의 설명을 풀어 쓴 문서를 `.html`(인쇄용 스타일)과 `.md`로 저장. 강조한 내용은 형광펜, 시험 포인트는 박스로 표시
- **GPU 가속** · Apple Silicon Mac은 MLX, NVIDIA는 CUDA, 인텔 GPU는 OpenVINO, AMD 등은 DirectML로 자동 실행합니다. 없으면 CPU(int8).
- **반복 인식 완화** · 같은 단어나 구절이 길게 반복되면(MLX, OpenVINO, faster-whisper 모두) 해당 구간만 다시 변환하고, 복구하지 못한 구간은 `[반복 인식 오류 · 재변환으로 복구하지 못함]`으로 표시합니다.

## 변환 엔진

`--engine`(명령줄) 또는 `engine`(MCP 도구 인자)으로 고를 수 있어요.

| 엔진 | 모델 | 실행기 | 노트북 CPU | 이럴 때 |
|---|---|---|---|---|
| `ko` (기본) | [batiai/batisay-ko-turbo](https://huggingface.co/batiai/batisay-ko-turbo) — Whisper large-v3-turbo를 한국어로 파인튜닝 | faster-whisper (int8) | 빠름 | 한국어 강의·회의 |
| `whisper` | OpenAI Whisper (`--model`로 `large-v3`, `large-v3-turbo` 등 선택) | faster-whisper (int8) | 빠름 | 영어 등 다른 언어, 여러 언어가 섞인 녹음 |
| `cohere` | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — 공개 당시 Open ASR 리더보드 1위 | transformers | 매우 느림, GPU 권장 | GPU가 있고 정확도를 최대로 원할 때 (`transformers>=5.4` 필요) |

**faster-whisper는 모델이 아니라 실행기**예요. Whisper 계열 모델을 CTranslate2 형식으로 바꿔서 CPU에서도 빠르게 돌려줘요. `ko` 엔진은 한국어 파인튜닝 모델을 `setup.bat`이 최초 1회 이 형식(int8)으로 변환해서 쓰기 때문에, 한국어 녹음에 맞춘 모델을 사용할 수 있어요.

### GPU 가속

| GPU | 방식 | 지원 엔진 | 설치 |
|---|---|---|---|
| Apple Silicon (M1 이상 Mac) | mlx-whisper (fp16, Metal) | `ko`, `whisper` | `install.py`(또는 `setup.sh`)가 자동으로 mlx-whisper를 설치하고 한국어 모델을 MLX 형식으로 1회 변환. `whisper` 엔진은 [mlx-community](https://huggingface.co/mlx-community) 변환본을 받아 씀 |
| NVIDIA (GeForce RTX/GTX) | faster-whisper + CUDA (float16) | `ko`, `whisper` | `setup.bat`이 `nvidia-smi`를 찾으면 cuBLAS·cuDNN을 pip로 설치 (CUDA Toolkit 불필요) |
| 인텔 내장(Iris Xe, Arc 내장) · Arc 외장 | OpenVINO GenAI (fp16) | `ko`, `whisper`의 `large-v3`·`large-v3-turbo` | `setup.bat`이 인텔 그래픽을 찾으면 OpenVINO를 설치하고 한국어 모델을 OpenVINO 형식으로 1회 변환 |
| AMD Radeon 등 DX12 GPU (Windows) | ONNX Runtime + DirectML | `ko`, `whisper`의 `large-v3`·`large-v3-turbo` | `install.py --amd-gpu` 또는 `setup.bat`(AMD 그래픽만 있을 때 자동). 모델을 ONNX로 1회 변환 |

자동 순서는 Apple Silicon(MLX) → NVIDIA → 인텔 GPU → DirectML(AMD 등) → CPU이고, GPU에서 시작이 안 되면 CPU로 넘어가요. `--cpu`로 CPU를 강제할 수 있어요. MLX·인텔 GPU·DirectML 경로는 긴 녹음을 조용한 틈에서 약 5분씩 잘라 처리하고 타임스탬프를 이어 붙여요. MLX 경로는 무음 구간을 건너뛰어 무음에서 헛말이 나오지 않게 해요.

어떤 경로가 실제로 쓰이는지는 `python transcribe.py --doctor`(Claude에서는 `check_setup` 도구)로 확인하세요. 장치 감지, 준비된 모델, 결과 저장 폴더를 보여 주고 10초 시험 음성을 그 경로로 실제 변환해 봐요.

> DirectML 경로는 AMD 실기기에서 아직 측정하지 못했어요. 같은 코드를 CPU 실행 장치로 바꿔 ONNX 변환·타임스탬프 변환·이어하기·반복 복구까지 검증했어요. AMD에서 돌려 보면 `--doctor` 결과를 이슈로 남겨 주세요. 인텔용 `optimum-intel`과 AMD용 `optimum-onnx`는 서로 다른 버전을 요구해서 한 가상환경에 같이 설치할 수 없어요.

- Intel GPU에서 일반 Whisper 모델을 처음 사용하면 `large-v3`는 약 3.1GB, `large-v3-turbo`는 약 1.6GB의 가중치를 내려받아 OpenVINO로 변환합니다. 이후에는 저장된 모델을 재사용합니다.
- 기존 결과를 다시 변환하려면 MCP에서 `force=true`, 명령줄에서는 `--force`를 지정하세요.
- `ko`, `whisper`는 문장마다 타임스탬프가 붙어요. `cohere`는 모델이 타임스탬프를 주지 않아서 녹음을 조용한 틈에서 약 20초씩 잘라 조각마다 변환하고, 조각 시작 시각을 붙여요. 덕분에 `cohere`도 이어하기와 반복 복구가 돼요.
- `cohere`는 언어 자동 감지가 없어 `--lang`/`language`를 맞게 줘야 하고, 추가 설치와 Hugging Face 약관 동의가 필요해요 (`transcribe.py` 맨 위 주석 참고).
- 정확도 차이는 녹음 상태에 따라 달라요. 결과는 엔진·모델마다 따로 저장되니 같은 녹음을 여러 모델로 돌려 직접 비교해보세요: `녹음.ko.txt`, `녹음.whisper.txt`(large-v3-turbo), `녹음.whisper.large-v3.txt`, `녹음.whisper.small.txt` 등.

### 노트북 실측

삼성 노트북(시스템 모델 `960XHA`), Intel Core Ultra 7 258V(8코어), Intel Arc 140V, RAM 32GB, Windows 11에서 측정했습니다. 같은 한국어 강의의 9:00–12:00 구간으로 첫 실행과 모델 재사용을 각각 한 번 측정했고, 모델 다운로드와 최초 형식 변환은 제외했습니다.

| 모델 | 장치 | 3분 녹음 첫 실행 | 3분 녹음 모델 재사용 | 2시간 환산 |
|---|---|---:|---:|---:|
| BatiSay KO Turbo | Intel Arc · OpenVINO FP16 | 17.00초 | 5.87초 | 약 4분 |
| Whisper `large-v3` | Intel Arc · OpenVINO FP16 | 57.56초 | 38.13초 | 약 25분 |
| Whisper `large-v3-turbo` | Intel Arc · OpenVINO FP16 | 19.83초 | 9.94초 | 약 7분 |
| Whisper `large-v3-turbo` | CPU · faster-whisper INT8 | 93.46초 | 105.46초 | 약 70분 |

2시간 환산은 모델을 불러둔 상태의 3분 측정값을 40배 한 값입니다. 별도의 실제 강의에서 `large-v3`는 **61분 11초를 7분 8초**에 변환했습니다(모델 로딩 포함, 같은 속도라면 2시간 약 14분). 녹음의 발화 밀도, 잡음, 재시도에 따라 실제 속도가 달라집니다. Cohere 엔진은 별도 모델 접근과 의존성이 필요해 측정하지 않았습니다.

## 설치와 실행

Python 3.10 이상이 필요합니다. Linux에서 가상환경을 만들 수 없다면 배포판의 Python venv 패키지를 설치하세요. Windows, macOS, Linux에서 저장소 폴더를 연 뒤 아래 명령으로 가상환경과 패키지를 설치합니다.

**Windows (PowerShell)**

```powershell
python install.py
.venv\Scripts\python.exe transcribe.py --choose-model
```

**macOS / Linux**

```bash
python3 install.py
.venv/bin/python transcribe.py --choose-model
```

녹음 파일 경로를 붙여넣고 모델 번호를 선택합니다. 파일 경로를 인자로 직접 넘길 수도 있습니다. Windows에서 기존 `setup.bat`과 `transcribe_menu.bat`도 사용할 수 있습니다.

- **Intel GPU** · x86-64 Windows/Linux에서 설치 명령에 `--intel-gpu`를 추가합니다. 호환 드라이버가 설치돼 있어야 하며, `ko`, `whisper large-v3`, `whisper large-v3-turbo`에서 사용합니다.
- **AMD GPU** · Windows에서 설치 명령에 `--amd-gpu`를 추가합니다(DirectML, `--intel-gpu`와 함께 쓸 수 없음). DX12를 지원하는 최신 그래픽 드라이버가 필요해요.
- **NVIDIA GPU** · x86-64 Windows/Linux에서 설치 명령에 `--nvidia-gpu`를 추가하면 CTranslate2용 cuBLAS·cuDNN을 pip로 설치합니다(CUDA Toolkit 불필요, 최신 드라이버 필요). Windows의 기존 `setup.bat`은 `nvidia-smi`가 있으면 자동으로 설치합니다.
- **모델 미리 준비** · `install.py`는 한국어 모델을 설치 중에 미리 내려받아 변환합니다(`--intel-gpu`면 OpenVINO 형식까지). 첫 변환이 멈춘 것처럼 보이지 않게 하려는 것이고, 나중으로 미루려면 `--skip-models`를 붙이세요.
- **macOS** · Apple Silicon(M1 이상)이면 `install.py`가 mlx-whisper를 자동으로 설치하고 한국어 모델을 MLX 형식으로 미리 변환해 GPU로 실행합니다. 인텔 Mac은 CPU 경로로 실행합니다. `./setup.sh`는 `install.py --register-claude`를 한 번에 실행하는 단축 스크립트예요.
- **Claude Desktop** · Windows/macOS에서 설치 명령에 `--register-claude`를 추가한 뒤 앱을 완전히 종료하고 다시 실행합니다.
- **Linux MCP 클라이언트** · stdio 서버로 등록합니다. 실행 명령에는 가상환경 Python의 절대 경로, 인자에는 `transcribe_mcp.py`의 절대 경로를 지정합니다.

## 결과 저장 위치

변환 결과, 삼성노트에서 꺼낸 파일, 정리본은 원본 옆이 아니라 **설치 폴더의 `output` 폴더** 한 곳에 녹음·노트별로 모여요. 원본이 있는 폴더(다운로드 등)에는 아무것도 만들지 않아요.

```
output/
├─ 운영체제 3주차/              ← 녹음 하나
│  ├─ source.txt               (원본 파일 경로)
│  ├─ 운영체제 3주차.ko.txt
│  └─ 운영체제 3주차.whisper.large-v3.txt
└─ Lecture12_…/                ← 삼성노트 하나
   ├─ 추출/                     (노트에서 꺼낸 녹음·PDF와 각 녹음의 변환 결과)
   └─ 정리본/                   (책 .html / .md)
```

- 다른 곳에 두려면 설치 폴더에 `config.json`을 만들어 `{"output_dir": "D:/강의결과"}`처럼 적거나, 환경 변수 `TRANSCRIBE_OUTPUT_DIR`을 설정하세요. `config.json`은 `.gitignore`에 들어 있어 개인 경로가 저장소에 올라가지 않아요.
- 이름이 같은 녹음이 다른 폴더에 있으면 `강의_1a2b3c`처럼 구분해서 겹치지 않아요.
- 예전 버전이 원본 옆에 만든 결과(`녹음.ko.txt`)가 있으면 새 위치로 복사해서 다시 변환하지 않고 그대로 써요.

## 사용 예 (Claude 데스크톱 앱)

- "다운로드 폴더에 있는 최근 녹음 요약해줘"
- "운영체제 3주차.m4a 텍스트로 바꿔서 시험 범위 위주로 정리해줘"
- "다운로드에 있는 Lecture12 노트, 녹음 바탕으로 책처럼 정리해줘"

## 도구

| 도구 | 하는 일 |
|---|---|
| `find_recordings` | 다운로드·바탕화면·문서 등에서 최근 녹음/영상/노트 찾기 |
| `start_transcription` / `get_transcription` | 변환 작업 시작 / 진행률·결과 조회 (긴 텍스트는 offset으로 나눠 받음) |
| `prepare_lecture_note` | .sdocx를 풀거나 PDF 옆의 녹음을 찾아 변환까지 시작 |
| `read_pdf_text` | PDF 페이지별 텍스트 |
| `save_book` | 원고를 `.md` + `.html`로 `output/<노트>/정리본`에 저장 (출력 폴더 밖에는 쓰지 않음) |
| `check_setup` | 설치·GPU 상태 점검과 10초 시험 변환 |

## 명령줄로 쓰기

Windows 예시입니다. macOS와 Linux에서는 Python 경로를 `.venv/bin/python`으로 바꿉니다.

```
.venv\Scripts\python transcribe.py 녹음.m4a
.venv\Scripts\python transcribe.py lecture.mp4 --engine whisper --model large-v3 --lang ko --gpu
```

## 테스트

모델 없이 핵심 동작(결과 저장 위치, 반복 인식 복구, 이어하기, Cohere 조각 타임스탬프, GPU 선택과 CPU 전환, 녹음 자르기, 삼성노트 풀기, 책 HTML, 자가 진단)을 확인합니다. GitHub에 push하면 Ubuntu·Windows에서 자동으로 돌아요.

```bash
pip install pytest
python -m pytest -q tests
```

## 구조

```
Claude (클라우드) ⇄ Claude 데스크톱 앱 ⇄ transcribe_mcp.py (stdio) → 변환 엔진(MLX·CUDA·OpenVINO·DirectML·CPU) → output/<녹음>/녹음.ko.txt
```

- `transcribe.py`: 변환 엔진 (CLI 겸용)
- `transcribe_mcp.py`: MCP 서버
- `lecture_tools.py`: 노트 풀기, PDF 텍스트, 책 HTML 생성
- `register_claude.py`: `claude_desktop_config.json`에 서버 등록 (기존 설정은 `.bak`으로 백업)

## 모델 라이선스

코드는 MIT지만, 내려받는 모델은 각자의 라이선스를 따라요.

- `batiai/batisay-ko-turbo` (기본 한국어 엔진): BatiAI Community 라이선스. 개인·연구용은 무료, 일정 규모 이상 상업적 이용은 별도 계약 필요
- Whisper (faster-whisper, mlx-whisper): MIT
- `CohereLabs/cohere-transcribe-03-2026`: Apache 2.0, Hugging Face 약관 동의 필요(게이트 모델)

## 한계

- 실제 삼성노트 파일 구조는 버전마다 다를 수 있어요. 녹음/PDF를 못 찾으면 앱에서 직접 내보내서 쓰세요.
- 그림 위주 슬라이드는 PDF에서 텍스트가 거의 안 나와서, 녹취 위주로 정리돼요.
