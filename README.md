# lecture-transcriber

로컬에서 강의 녹음을 텍스트로 바꾸고, Claude 데스크톱 앱이 그 결과로 요약하거나 **강의 노트(삼성노트·PDF)를 책처럼 정리**할 수 있게 해주는 MCP 서버예요.

> Local speech-to-text MCP server for the Claude desktop app (Windows). Transcribes lecture recordings on your own machine and lets Claude turn Samsung Notes / PDF slides + recordings into a book-style study guide.

Claude는 오디오를 직접 입력으로 받지 못해서, 음성 인식은 내 PC에서 돌리고 Claude에게는 **텍스트만** 넘기는 구조예요. 녹음 파일은 PC 밖으로 나가지 않아요.

## 기능

- **녹음 → 텍스트**: 한국어 파인튜닝 Whisper(기본), 다국어 Whisper, Cohere Transcribe 중 선택 ([변환 엔진](#변환-엔진) 참고)
- **긴 녹음 대응**: 백그라운드 작업 + 진행률 조회로 도구 호출 타임아웃 회피, 결과 캐시
- **삼성노트(.sdocx) 처리**: 압축을 풀고 파일 시그니처로 녹음·PDF를 골라냄
- **책 만들기**: 슬라이드 순서를 뼈대로 강의 설명을 풀어 쓴 문서를 `.html`(인쇄용 스타일)과 `.md`로 저장. 강조한 내용은 형광펜, 시험 포인트는 박스로 표시
- GPU 자동 사용: NVIDIA는 CUDA, 인텔 내장 그래픽(Iris Xe 등)·Arc는 OpenVINO, 없으면 CPU(int8)

## 변환 엔진

`--engine`(명령줄) 또는 `engine`(MCP 도구 인자)으로 고를 수 있어요.

| 엔진 | 모델 | 실행기 | 노트북 CPU | 이럴 때 |
|---|---|---|---|---|
| `ko` (기본) | [batiai/batisay-ko-turbo](https://huggingface.co/batiai/batisay-ko-turbo) — Whisper large-v3-turbo를 한국어로 파인튜닝 | faster-whisper (int8) | 빠름 | 한국어 강의·회의 |
| `whisper` | OpenAI Whisper (`--model`로 tiny ~ large-v3-turbo 선택) | faster-whisper (int8) | 빠름 | 영어 등 다른 언어, 여러 언어가 섞인 녹음 |
| `cohere` | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — 공개 당시 Open ASR 리더보드 1위 | transformers | 매우 느림, GPU 권장 | GPU가 있고 정확도를 최대로 원할 때 |

**faster-whisper는 모델이 아니라 실행기**예요. Whisper 계열 모델을 CTranslate2 형식으로 바꿔서 CPU에서도 빠르게 돌려줘요. `ko` 엔진은 한국어 파인튜닝 모델을 `setup.bat`이 최초 1회 이 형식(int8)으로 변환해서 쓰기 때문에, 일반 Whisper와 같은 속도로 한국어 정확도를 더 얻을 수 있어요.

### GPU 가속

| GPU | 방식 | 지원 엔진 | 설치 |
|---|---|---|---|
| NVIDIA (GeForce RTX/GTX) | faster-whisper + CUDA (float16) | `ko`, `whisper` | `setup.bat`이 `nvidia-smi`를 찾으면 cuBLAS·cuDNN을 pip로 설치 (CUDA Toolkit 불필요) |
| 인텔 내장(Iris Xe, Arc 내장) · Arc 외장 | OpenVINO GenAI (fp16) | `ko` | `setup.bat`이 인텔 그래픽을 찾으면 OpenVINO를 설치하고 한국어 모델을 OpenVINO 형식으로 1회 변환 |
| AMD Radeon | 아직 미지원 (CPU로 동작) | — | — |

자동 순서는 NVIDIA → 인텔 GPU → CPU이고, GPU에서 시작이 안 되면 CPU로 넘어가요. `--cpu`로 CPU를 강제할 수 있어요. 인텔 GPU 경로는 긴 녹음을 조용한 틈에서 약 5분씩 잘라 처리하고 타임스탬프를 이어 붙여요.

- `ko`, `whisper`는 NVIDIA GPU를 자동으로 써요. `setup.bat`이 `nvidia-smi`를 찾으면 cuBLAS·cuDNN을 pip로 함께 설치하니 CUDA Toolkit을 따로 깔 필요는 없어요. GPU 드라이버만 최신이면 돼요. 강제로 CPU를 쓰려면 `--cpu`. 인텔·AMD 그래픽은 지원하지 않아요.
- `ko`, `whisper`는 타임스탬프가 붙고, `cohere`는 타임스탬프 없이 문장 단위로만 나와요.
- `cohere`는 언어 자동 감지가 없어 `--lang`/`language`를 맞게 줘야 하고, 추가 설치와 Hugging Face 약관 동의가 필요해요 (`transcribe.py` 맨 위 주석 참고).
- 정확도 차이는 녹음 상태에 따라 달라요. 같은 녹음을 엔진별로 돌리면 `녹음.ko.txt`, `녹음.whisper.txt`처럼 따로 저장되니 직접 비교해보세요.

## 설치 (Windows)

1. Python 3.10+ 설치 (설치할 때 "Add python.exe to PATH" 체크)
2. 이 저장소를 받아서 오래 둘 폴더에 둬요 (예: `C:\tools\lecture-transcriber`)
3. `setup.bat` 실행 — 가상환경, 패키지, 한국어 모델 변환, Claude 데스크톱 설정 등록까지 자동 (최초 10~15분)
4. Claude 데스크톱 앱을 트레이에서 완전히 종료 후 다시 실행

인터넷에서 받은 `.bat`이라 SmartScreen이 막으면 "추가 정보 → 실행"을 누르거나, zip 파일 속성에서 "차단 해제"를 체크한 뒤 압축을 풀어요.

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
| `save_book` | 원고를 `.md` + `.html`로 저장 (처리한 노트 폴더에만 쓰기 허용) |

## 명령줄로 쓰기

```
.venv\Scripts\python transcribe.py 녹음.m4a
.venv\Scripts\python transcribe.py lecture.mp4 --engine whisper --lang en
```

## 구조

```
Claude (클라우드) ⇄ Claude 데스크톱 앱 ⇄ transcribe_mcp.py (stdio) → faster-whisper → 녹음.ko.txt
```

- `transcribe.py`: 변환 엔진 (CLI 겸용)
- `transcribe_mcp.py`: MCP 서버
- `lecture_tools.py`: 노트 풀기, PDF 텍스트, 책 HTML 생성
- `register_claude.py`: `claude_desktop_config.json`에 서버 등록 (기존 설정은 `.bak`으로 백업)

## 모델 라이선스

코드는 MIT지만, 내려받는 모델은 각자의 라이선스를 따라요.

- `batiai/batisay-ko-turbo` (기본 한국어 엔진): BatiAI Community 라이선스. 개인·연구용은 무료, 일정 규모 이상 상업적 이용은 별도 계약 필요
- Whisper (faster-whisper): MIT
- `CohereLabs/cohere-transcribe-03-2026`: Apache 2.0, Hugging Face 약관 동의 필요(게이트 모델)

## 한계

- 실제 삼성노트 파일 구조는 버전마다 다를 수 있어요. 녹음/PDF를 못 찾으면 앱에서 직접 내보내서 쓰세요.
- 그림 위주 슬라이드는 PDF에서 텍스트가 거의 안 나와서, 녹취 위주로 정리돼요.
