"""
녹음 파일 → 텍스트 변환 (로컬 실행, 엔진 3종 선택)
명령줄에서 직접 써도 되고, transcribe_mcp.py가 이 파일을 불러서 Claude에 연결해요.

엔진
  ko      : batiai/batisay-ko-turbo (Whisper turbo 한국어 파인튜닝, 타임스탬프 O) ← 한국어 추천
  whisper : faster-whisper 기본 모델 (다국어, 타임스탬프 O)
  cohere  : CohereLabs/cohere-transcribe-03-2026 (오픈 ASR 리더보드 1위, 약 20초 단위 타임스탬프, GPU 권장)

사용
  python transcribe.py 녹음.m4a                       # ko 엔진
  python transcribe.py 녹음.m4a --engine whisper --model large-v3-turbo --lang en
  python transcribe.py 녹음.m4a --engine cohere
  (GPU 자동 사용: Apple Silicon Mac → MLX, NVIDIA → CUDA, 인텔 내장/Arc → OpenVINO,
   그 밖의 DX12 GPU(AMD 등) → DirectML, 없으면 CPU. --cpu로 끌 수 있음)
결과
  녹음.<engine>.txt 가 같은 폴더에 생성

cohere 엔진은 추가 설치가 필요해요:
  pip install "transformers>=5.4.0" accelerate librosa soundfile sentencepiece protobuf
  + https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 에서 약관 동의 후 `hf auth login`
"""
import argparse
import glob
import os
import re
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
KO_REPO = "batiai/batisay-ko-turbo"
KO_DIR = HERE / "models" / "batisay-ko-turbo-ct2"

ENGINES = ("ko", "whisper", "cohere")

# 진행 메시지는 전부 stderr로 (MCP는 stdout을 통신용으로 쓰기 때문)
def log(*a):
    print(*a, file=sys.stderr, flush=True)


def fmt(t) -> str:
    if t is None:
        return "--:--:--"
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- 모델 로딩 (한 번 로드하면 재사용) ----------
_models = {}
_models_lock = threading.Lock()


def _add_cuda_dll_dirs():
    """pip로 설치한 NVIDIA 라이브러리(cuBLAS, cuDNN)를 Windows가 찾을 수 있게 경로 등록.
    CUDA Toolkit을 따로 설치하지 않아도 GPU를 쓸 수 있게 하려는 것."""
    try:
        import nvidia  # nvidia-cublas-cu12 / nvidia-cudnn-cu12 패키지
    except ImportError:
        return
    for base in getattr(nvidia, "__path__", []):
        for sub in Path(base).glob("*/bin"):
            try:
                os.add_dll_directory(str(sub))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")


def cuda_available() -> bool:
    _add_cuda_dll_dirs()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _resolve_device(gpu):
    """gpu: True=GPU 강제, False=CPU 강제, None=자동(있으면 GPU)."""
    if gpu is False:
        return "cpu"
    if cuda_available():
        return "cuda"
    if gpu:
        log("경고: GPU를 요청했지만 CUDA를 쓸 수 없어 CPU로 진행해요.")
    return "cpu"


def _whisper_model(name_or_path, gpu=None):
    from faster_whisper import WhisperModel

    device = _resolve_device(gpu)
    key = ("fw", name_or_path, device)
    with _models_lock:
        if key not in _models:
            log(f"모델 로드 중: {name_or_path} ({device})")
            if device == "cuda":
                try:
                    _models[key] = WhisperModel(name_or_path, device="cuda", compute_type="float16")
                    return _models[key]
                except Exception as e:  # noqa: BLE001  (드라이버·cuDNN 문제 등)
                    log(f"GPU로 모델을 올리지 못해 CPU로 바꿔요: {type(e).__name__}: {e}")
                    key = ("fw", name_or_path, "cpu")
                    if key in _models:
                        return _models[key]
            _models[key] = WhisperModel(
                name_or_path, device="cpu", compute_type="int8",
                cpu_threads=os.cpu_count() or 4,  # CPU 코어 전부 사용
            )
        return _models[key]


def ensure_ko_model():
    """한국어 파인튜닝 모델을 faster-whisper용(CTranslate2 int8)으로 최초 1회 변환.
    변환에만 torch/transformers가 필요하고, 이후 실행은 faster-whisper만 씀."""
    if (KO_DIR / "model.bin").exists():
        return str(KO_DIR)

    log(f"[최초 1회] {KO_REPO} 를 faster-whisper 형식으로 변환합니다 (몇 분 걸림)...")
    try:
        from ctranslate2.converters import TransformersConverter
        from huggingface_hub import list_repo_files
    except ImportError as e:
        raise RuntimeError(
            "한국어 모델 변환에 필요한 패키지가 없어요. setup.bat(Windows)/setup.sh(macOS)를 다시 실행하거나:\n"
            "  pip install ctranslate2 transformers torch huggingface_hub"
        ) from e

    # faster-whisper가 쓰는 토크나이저·전처리 설정 중 레포에 있는 것만 복사
    wanted = ["tokenizer.json", "preprocessor_config.json"]
    present = set(list_repo_files(KO_REPO))
    copy = [f for f in wanted if f in present]
    if "preprocessor_config.json" not in copy:
        log("경고: preprocessor_config.json이 없어 결과가 이상할 수 있어요.")

    KO_DIR.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(KO_REPO, copy_files=copy).convert(
        str(KO_DIR), quantization="int8", force=True
    )
    log(f"변환 완료: {KO_DIR}")
    return str(KO_DIR)


# ---------- 엔진 ----------
# 모든 엔진은 (시작초 또는 None, 텍스트)를 yield 하고,
# on_progress(0~1)를 받으면 진행률을 알려줌.

SR = 16000


def _decode_from(audio, start_at):
    """start_at초부터의 오디오(16kHz float32)와 전체 길이(초)."""
    from faster_whisper.audio import decode_audio
    wav = decode_audio(audio, sampling_rate=SR)
    return wav[int(start_at * SR):], len(wav) / SR


def run_whisper(audio, lang="ko", model="large-v3-turbo", gpu=None, on_progress=None, start_at=0.0):
    m = _whisper_model(model, gpu)
    # 반복 구간을 다시 변환하려면 원본 파형이 필요해서 한 번만 디코딩해 둠
    # (faster-whisper도 내부에서 어차피 전체를 디코딩하므로 메모리 사용은 같음)
    wav, total = _decode_from(audio, start_at)
    segments, _ = m.transcribe(wav, language=lang, vad_filter=True)
    for seg in segments:
        if on_progress and total:
            on_progress(min((start_at + seg.end) / total, 1.0))
        text = seg.text.strip()
        if _has_transcription_loop(text):
            text = _retry_whisper_loop(m, wav, seg.start, seg.end, lang)
        yield start_at + seg.start, text


def _retry_whisper_loop(model, wav, start, end, lang):
    """faster-whisper가 같은 구절을 반복 출력한 구간만, 앞 문맥 없이 다시 변환."""
    clip = wav[int(start * SR):int(min(end, start + 25.0) * SR)]
    if len(clip):
        try:
            segs, _ = model.transcribe(clip, language=lang, vad_filter=False,
                                       condition_on_previous_text=False,
                                       temperature=(0.2, 0.4, 0.6))
            text = " ".join(s.text.strip() for s in segs).strip()
            if text and not _has_transcription_loop(text):
                return text
        except RuntimeError as exc:
            log(f"반복 인식 구간 재시도 실패: {exc}")
    return LOOP_MARKER


def gpu_backend(gpu=None) -> str:
    """쓸 가속 방식. 자동 순서: Apple Silicon(MLX) → NVIDIA(CUDA) → 인텔 GPU(OpenVINO)
    → 그 밖의 DX12 GPU(DirectML, AMD 등) → CPU. gpu=False면 항상 'cpu'."""
    if gpu is False:
        return "cpu"
    if mlx_available():
        return "mlx"
    if cuda_available():
        return "cuda"
    if intel_gpu_available():
        return "openvino"
    if directml_available():
        return "directml"
    return "cpu"


def run_ko(audio, gpu=None, on_progress=None, start_at=0.0, **_):
    backend = gpu_backend(gpu)
    try:
        if backend == "mlx":
            yield from run_mlx(audio, ensure_ko_mlx_model(), "ko", on_progress=on_progress,
                               start_at=start_at)
            return
        if backend == "openvino":
            yield from run_openvino(audio, ensure_ko_ov_model(), "ko", on_progress=on_progress,
                                    start_at=start_at)
            return
        if backend == "directml":
            yield from run_onnx(audio, ensure_onnx_model(KO_REPO, KO_ONNX_DIR), "ko",
                                on_progress=on_progress, start_at=start_at)
            return
    except _GPUStartError as e:
        log(f"GPU({backend})로 시작하지 못해 CPU로 바꿔요: {e}")
    yield from run_whisper(audio, "ko", ensure_ko_model(), gpu, on_progress, start_at)


# ---------- 인텔 GPU (OpenVINO) ----------
KO_OV_DIR = HERE / "models" / "batisay-ko-turbo-ov"
LARGE_V3_OV_DIR = HERE / "models" / "whisper-large-v3-ov"
LARGE_V3_TURBO_OV_DIR = HERE / "models" / "whisper-large-v3-turbo-ov"
_WHISPER_OV_DIRS = {
    "large-v3": LARGE_V3_OV_DIR,
    "large-v3-turbo": LARGE_V3_TURBO_OV_DIR,
    "turbo": LARGE_V3_TURBO_OV_DIR,
}


class _GPUStartError(RuntimeError):
    """GPU 경로로 시작조차 못 했을 때(설치·변환·장치 문제). 이미 텍스트를 내보낸 뒤의 오류와 구분.
    이 예외가 나면 CPU로 넘어감."""


class _OVStartError(_GPUStartError):
    """OpenVINO(인텔 GPU) 시작 실패."""


def ov_device():
    """OpenVINO가 쓸 수 있는 인텔 GPU 이름 ('GPU' 또는 'GPU.0'), 없으면 None. AMD·NVIDIA는 여기 안 잡힘."""
    try:
        import openvino as ov
        devs = ov.Core().available_devices
    except Exception:  # noqa: BLE001
        return None
    gpus = [d for d in devs if d.startswith("GPU")]
    if not gpus:
        return None
    # 내장+외장이 같이 있으면 GPU.1이 보통 외장(Arc)
    return sorted(gpus)[-1] if len(gpus) > 1 else gpus[0]


def intel_gpu_available() -> bool:
    if os.environ.get("TRANSCRIBE_NO_OPENVINO"):
        return False
    return ov_device() is not None


def ensure_ko_ov_model():
    return _ensure_ov_model(KO_REPO, KO_OV_DIR)


def _ensure_ov_tokenizers(model_dir: Path) -> None:
    names = ("openvino_tokenizer", "openvino_detokenizer")
    if all((model_dir / (name + ext)).is_file()
           for name in names for ext in (".xml", ".bin")):
        return
    try:
        import openvino as ov
        from openvino_tokenizers import convert_tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        models = convert_tokenizer(tokenizer, with_detokenizer=True)
        for name, model in zip(names, models, strict=True):
            ov.save_model(model, model_dir / (name + ".xml"))
    except Exception as exc:
        raise _OVStartError(f"OpenVINO 토크나이저 변환 실패: {exc}") from exc


def _clean_stale_exports(parent: Path, max_age_s: float = 3 * 3600) -> None:
    """변환이 중간에 강제 종료되면 models/tmpXXXX 폴더가 남아 수 GB를 차지함. 오래된 것만 지움
    (다른 프로세스가 지금 변환 중인 폴더는 최근에 파일이 써지므로 건드리지 않음)."""
    import shutil
    import time

    if not parent.is_dir():
        return
    now = time.time()
    for d in parent.glob("tmp*"):
        if not d.is_dir():
            continue
        try:
            newest = max((f.stat().st_mtime for f in d.rglob("*")), default=d.stat().st_mtime)
        except OSError:
            continue
        if now - newest > max_age_s:
            log(f"중단된 변환 임시 폴더 정리: {d}")
            shutil.rmtree(d, ignore_errors=True)


def _ensure_ov_model(repo: str, model_dir: Path) -> str:
    """Export once, then publish only the completed OpenVINO model."""
    import subprocess
    import tempfile

    with _models_lock:
        if (model_dir / "openvino_encoder_model.xml").is_file():
            _ensure_ov_tokenizers(model_dir)
            return str(model_dir)
        log(f"[최초 1회] {repo} 를 인텔 GPU(OpenVINO) 형식으로 변환합니다.")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        _clean_stale_exports(model_dir.parent)
        with tempfile.TemporaryDirectory(dir=model_dir.parent) as temp:
            staging = Path(temp) / "model"
            cmd = [sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino",
                   "--model", repo, "--task", "automatic-speech-recognition-with-past",
                   "--weight-format", "fp16", str(staging)]
            result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
            if result.returncode != 0 or not (staging / "openvino_encoder_model.xml").is_file():
                raise _OVStartError(f"{repo} OpenVINO 변환 실패")
            _ensure_ov_tokenizers(staging)
            try:
                staging.rename(model_dir)
            except FileExistsError:
                if not (model_dir / "openvino_encoder_model.xml").is_file():
                    raise _OVStartError(f"완료되지 않은 모델 폴더가 있습니다: {model_dir}")
        log(f"변환 완료: {model_dir}")
        return str(model_dir)

def _speech_spans(wav, sr):
    """말소리 구간 [(시작초, 끝초), ...]. VAD를 못 쓰면 None."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        speech = get_speech_timestamps(wav, VadOptions(min_silence_duration_ms=300))
    except Exception:  # noqa: BLE001
        return None
    return [(s["start"] / sr, s["end"] / sr) for s in speech]


def _cut_points(wav, sr, target_s=300, spans=None):
    """긴 녹음을 약 target_s초 단위로 자르되, 말하는 중간이 아니라 조용한 틈에서 자름.
    spans(말소리 구간)를 이미 구했으면 넘겨서 VAD를 다시 돌리지 않게 할 수 있음."""
    total = len(wav) / sr
    if total <= target_s * 1.2:
        return [0.0, total]
    if spans is None:
        spans = _speech_spans(wav, sr) or []
    gaps = [(a[1] + b[0]) / 2 for a, b in zip(spans, spans[1:])]
    # 목표 지점 앞뒤 window초 안의 조용한 틈 중 가장 가까운 곳에서 자르고, 조각은 최소 min_len초
    window, min_len = min(30.0, target_s * 0.4), min(30.0, target_s * 0.5)
    cuts, t = [0.0], target_s
    while t < total - target_s * 0.2:
        near = [g for g in gaps if abs(g - t) < window and g > cuts[-1] + min_len]
        c = min(near, key=lambda g: abs(g - t)) if near else t
        cuts.append(c)
        t = c + target_s
    cuts.append(total)
    return cuts


_REPEATED_PHRASE = re.compile(
    r"(?<!\S)(\S+(?:\s+\S+){0,5}?)(?:\s+\1){5,}(?=\s|$)"
)


LOOP_MARKER = "[반복 인식 오류 · 재변환으로 복구하지 못함]"


def _has_transcription_loop(text: str) -> bool:
    # 짧은 숫자열(예: 비트 스터핑 예제)과 일반적인 반복은 유지한다.
    return any(len(match[0]) >= 35 for match in _REPEATED_PHRASE.finditer(text))


def _openvino_rows(pipe, audio, lang):
    result = pipe.generate(audio.tolist(), language=f"<|{lang}|>", task="transcribe",
                           return_timestamps=True)
    chunks = result.chunks or []
    if chunks and isinstance(chunks[0], (list, tuple)):
        chunks = chunks[0]
    if chunks:
        return [(max(chunk.start_ts, 0.0), chunk.text.strip()) for chunk in chunks]
    text = (result.texts[0] if getattr(result, "texts", None) else str(result)).strip()
    return [(0.0, text)] if text else []


def _retry_openvino_loops(pipe, piece, sr, lang, rows):
    return _retry_loops(lambda clip: _openvino_rows(pipe, clip, lang), piece, sr, rows)


def _retry_loops(transcribe_clip, piece, sr, rows):
    """같은 구절이 길게 반복된 줄만 그 구간(최대 25초)을 다시 변환. 그래도 반복이면 표시."""
    for index, (offset, text) in enumerate(rows):
        if not text:
            continue
        if _has_transcription_loop(text):
            next_offset = rows[index + 1][0] if index + 1 < len(rows) else len(piece) / sr
            end = min(len(piece) / sr, max(offset + 2.0, next_offset), offset + 25.0)
            clip = piece[int(offset * sr):int(end * sr)]
            if len(clip):
                try:
                    text = " ".join(retry for _, retry in transcribe_clip(clip)).strip()
                except RuntimeError as exc:
                    log(f"반복 인식 구간 재시도 실패: {exc}")
                    text = ""
            if not len(clip) or not text or _has_transcription_loop(text):
                text = LOOP_MARKER
        yield offset, text


def run_openvino(audio, model_dir, lang="ko", device=None, on_progress=None, start_at=0.0):
    try:
        import openvino_genai as ovg
        from faster_whisper.audio import decode_audio
        device = device or ov_device() or "CPU"
        key = ("ov", model_dir, device)
        with _models_lock:
            if key not in _models:
                log(f"모델 로드 중: {model_dir} (OpenVINO {device})")
                Pipe = getattr(ovg, "ASRPipeline", None) or ovg.WhisperPipeline
                _models[key] = Pipe(model_dir, device)
        pipe = _models[key]
        sr = SR
        wav, full = _decode_from(audio, start_at)
    except Exception as e:  # noqa: BLE001
        raise _OVStartError(f"{type(e).__name__}: {e}") from e

    cuts = _cut_points(wav, sr)
    total = full
    for a, b in zip(cuts, cuts[1:]):
        piece = wav[int(a * sr):int(b * sr)]
        rows = _openvino_rows(pipe, piece, lang)
        for offset, text in _retry_openvino_loops(pipe, piece, sr, lang, rows):
            yield start_at + a + offset, text
        if on_progress and total:
            on_progress(min((start_at + b) / total, 1.0))


def _run_whisper_openvino(audio, lang, model, gpu, on_progress, start_at):
    repo_model = "large-v3-turbo" if model == "turbo" else model
    try:
        model_dir = _ensure_ov_model(f"openai/whisper-{repo_model}", _WHISPER_OV_DIRS[model])
        yield from run_openvino(audio, model_dir, lang, on_progress=on_progress, start_at=start_at)
        return
    except _OVStartError as exc:
        log(f"{model} 인텔 GPU 시작 실패, CPU로 전환: {exc}")
    yield from run_whisper(audio, lang, model, gpu, on_progress, start_at)


# ---------- 그 밖의 GPU: DirectML (AMD Radeon 등, Windows) ----------
# DirectML은 DX12가 되는 GPU면 제조사와 상관없이 쓸 수 있는 Microsoft의 가속 방식.
# 모델을 ONNX로 한 번 변환해 두고 ONNX Runtime(onnxruntime-directml)으로 실행한다.
KO_ONNX_DIR = HERE / "models" / "batisay-ko-turbo-onnx"
_WHISPER_ONNX_DIRS = {
    "large-v3": HERE / "models" / "whisper-large-v3-onnx",
    "large-v3-turbo": HERE / "models" / "whisper-large-v3-turbo-onnx",
    "turbo": HERE / "models" / "whisper-large-v3-turbo-onnx",
}
ONNX_PIECE_S = 300


def _ort_provider() -> str:
    # 테스트·진단용으로 실행 장치를 바꿀 수 있게 (예: CPUExecutionProvider)
    return os.environ.get("TRANSCRIBE_ORT_PROVIDER", "DmlExecutionProvider")


def directml_available() -> bool:
    if os.environ.get("TRANSCRIBE_NO_DIRECTML"):
        return False
    try:
        import onnxruntime as ort
        import optimum.onnxruntime  # noqa: F401  (설치돼 있어야 이 경로를 씀)
    except Exception:  # noqa: BLE001
        return False
    return _ort_provider() in ort.get_available_providers()


def ensure_onnx_model(repo: str, model_dir: Path) -> str:
    """Hugging Face 모델을 ONNX로 한 번 변환. 임시 폴더에서 끝까지 만든 뒤에만 옮겨서 반쯤 된 폴더가 안 남음."""
    import subprocess
    import tempfile

    done = model_dir / "encoder_model.onnx"
    with _models_lock:
        if done.is_file():
            return str(model_dir)
        log(f"[최초 1회] {repo} 를 ONNX(DirectML) 형식으로 변환합니다.")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        _clean_stale_exports(model_dir.parent)
        with tempfile.TemporaryDirectory(dir=model_dir.parent) as temp:
            staging = Path(temp) / "model"
            cmd = [sys.executable, "-m", "optimum.commands.optimum_cli", "export", "onnx",
                   "--model", repo, "--task", "automatic-speech-recognition-with-past", str(staging)]
            result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
            if result.returncode != 0 or not (staging / "encoder_model.onnx").is_file():
                raise _GPUStartError(f"{repo} ONNX 변환 실패")
            # 변환기가 Whisper 전용 생성 설정(no_timestamps_token_id, lang_to_id 등)을 빠뜨리는 경우가 있어
            # 원본 설정을 그대로 덮어씀. 이게 없으면 타임스탬프를 요청할 때 오류가 남.
            try:
                from transformers import GenerationConfig
                GenerationConfig.from_pretrained(repo).save_pretrained(staging)
            except Exception as exc:  # noqa: BLE001
                raise _GPUStartError(f"{repo} 생성 설정 복사 실패: {exc}") from exc
            try:
                staging.rename(model_dir)
            except FileExistsError:
                if not done.is_file():
                    raise _GPUStartError(f"완료되지 않은 모델 폴더가 있습니다: {model_dir}")
        log(f"변환 완료: {model_dir}")
        return str(model_dir)


def _onnx_pipeline(model_dir: str):
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline

    provider = _ort_provider()
    key = ("onnx", model_dir, provider)
    with _models_lock:
        if key not in _models:
            log(f"모델 로드 중: {model_dir} (ONNX Runtime {provider})")
            options = ort.SessionOptions()
            # DirectML은 메모리 패턴 최적화와 병렬 실행을 지원하지 않음
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            model = ORTModelForSpeechSeq2Seq.from_pretrained(
                model_dir, provider=provider, session_options=options, use_io_binding=False)
            processor = AutoProcessor.from_pretrained(model_dir)
            _models[key] = pipeline("automatic-speech-recognition", model=model,
                                    tokenizer=processor.tokenizer,
                                    feature_extractor=processor.feature_extractor,
                                    chunk_length_s=30)
        return _models[key]


def _onnx_rows(pipe, audio, lang):
    out = pipe({"raw": audio, "sampling_rate": SR}, return_timestamps=True,
               generate_kwargs={"language": lang, "task": "transcribe"})
    rows = [((c.get("timestamp") or (0.0, None))[0] or 0.0, c["text"].strip())
            for c in out.get("chunks") or [] if c["text"].strip()]
    if not rows and out.get("text", "").strip():
        rows = [(0.0, out["text"].strip())]
    return rows


def run_onnx(audio, model_dir, lang="ko", on_progress=None, start_at=0.0):
    try:
        pipe = _onnx_pipeline(model_dir)
        wav, total = _decode_from(audio, start_at)
    except Exception as e:  # noqa: BLE001
        raise _GPUStartError(f"{type(e).__name__}: {e}") from e
    cuts = _cut_points(wav, SR, target_s=ONNX_PIECE_S)
    for a, b in zip(cuts, cuts[1:]):
        piece = wav[int(a * SR):int(b * SR)]
        rows = _onnx_rows(pipe, piece, lang)
        for offset, text in _retry_loops(lambda clip: _onnx_rows(pipe, clip, lang), piece, SR, rows):
            yield start_at + a + offset, text
        if on_progress and total:
            on_progress(min((start_at + b) / total, 1.0))


def _run_whisper_onnx(audio, lang, model, gpu, on_progress, start_at):
    repo_model = "large-v3-turbo" if model == "turbo" else model
    try:
        model_dir = ensure_onnx_model(f"openai/whisper-{repo_model}", _WHISPER_ONNX_DIRS[model])
        yield from run_onnx(audio, model_dir, lang, on_progress=on_progress, start_at=start_at)
        return
    except _GPUStartError as exc:
        log(f"{model} DirectML 시작 실패, CPU로 전환: {exc}")
    yield from run_whisper(audio, lang, model, gpu, on_progress, start_at)


# ---------- Apple Silicon (MLX) ----------
KO_MLX_DIR = HERE / "models" / "batisay-ko-turbo-mlx"
# whisper 엔진 --model 이름 → mlx-community 변환본
MLX_WHISPER_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class _MLXStartError(_GPUStartError):
    """MLX(Apple GPU) 시작 실패."""


def mlx_available() -> bool:
    """Apple Silicon Mac이고 mlx-whisper가 설치돼 있으면 True."""
    if sys.platform != "darwin" or os.environ.get("TRANSCRIBE_NO_MLX"):
        return False
    import platform
    if platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _hf_to_mlx_name(k):
    """Hugging Face Whisper 가중치 이름 → mlx-whisper(OpenAI 형식) 이름. 안 쓰는 건 None."""
    if k.startswith("model."):
        k = k[len("model."):]
    if k in ("encoder.embed_positions.weight", "proj_out.weight"):
        return None  # 인코더 위치 임베딩은 mlx가 직접 계산, 출력층은 토큰 임베딩과 공유
    for a, b in [
        ("decoder.embed_tokens.", "decoder.token_embedding."),
        ("decoder.embed_positions.weight", "decoder.positional_embedding"),
        ("encoder.layer_norm.", "encoder.ln_post."),
        ("decoder.layer_norm.", "decoder.ln."),
        (".layers.", ".blocks."),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".self_attn.", ".attn."),
        (".encoder_attn.", ".cross_attn."),
        (".q_proj.", ".query."),
        (".k_proj.", ".key."),
        (".v_proj.", ".value."),
        (".out_proj.", ".out."),
        (".fc1.", ".mlp1."),
        (".fc2.", ".mlp2."),
    ]:
        k = k.replace(a, b)
    return k


def ensure_ko_mlx_model():
    """한국어 모델을 MLX 형식(fp16)으로 최초 1회 변환. torch 없이 safetensors만으로 변환.
    임시 폴더에서 끝까지 만든 뒤에만 옮겨서 반쯤 된 폴더가 안 남음."""
    import json
    import shutil
    import tempfile

    with _models_lock:
        if (KO_MLX_DIR / "weights.safetensors").is_file():
            return str(KO_MLX_DIR)
        log(f"[최초 1회] {KO_REPO} 를 Apple GPU(MLX) 형식으로 변환합니다...")
        try:
            import mlx.core as mx
            from huggingface_hub import hf_hub_download

            cfg = json.loads(Path(hf_hub_download(KO_REPO, "config.json")).read_text(encoding="utf-8"))
            src = mx.load(hf_hub_download(KO_REPO, "model.safetensors"))
            weights = {}
            for k, v in src.items():
                name = _hf_to_mlx_name(k)
                if name is None:
                    continue
                if name.endswith(("conv1.weight", "conv2.weight")):
                    v = v.swapaxes(1, 2)  # torch (out, in, k) → mlx (out, k, in)
                weights[name] = v.astype(mx.float16)
            dims = {
                "n_mels": cfg["num_mel_bins"],
                "n_audio_ctx": cfg["max_source_positions"],
                "n_audio_state": cfg["d_model"],
                "n_audio_head": cfg["encoder_attention_heads"],
                "n_audio_layer": cfg["encoder_layers"],
                "n_vocab": cfg["vocab_size"],
                "n_text_ctx": cfg["max_target_positions"],
                "n_text_state": cfg["d_model"],
                "n_text_head": cfg["decoder_attention_heads"],
                "n_text_layer": cfg["decoder_layers"],
            }
            KO_MLX_DIR.parent.mkdir(parents=True, exist_ok=True)
            _clean_stale_exports(KO_MLX_DIR.parent)
            with tempfile.TemporaryDirectory(dir=KO_MLX_DIR.parent) as temp:
                staging = Path(temp) / "model"
                staging.mkdir()
                (staging / "config.json").write_text(
                    json.dumps({**dims, "model_type": "whisper"}, indent=2), encoding="utf-8")
                mx.save_safetensors(str(staging / "weights.safetensors"), weights)
                if KO_MLX_DIR.exists():
                    shutil.rmtree(KO_MLX_DIR)  # 예전에 덜 만들어진 폴더
                staging.rename(KO_MLX_DIR)
        except Exception as e:  # noqa: BLE001
            raise _MLXStartError(f"MLX 변환 실패: {type(e).__name__}: {e}") from e
        log(f"변환 완료: {KO_MLX_DIR}")
        return str(KO_MLX_DIR)


def _clips_in(spans, a, b, pad=0.2, join=3.0):
    """[a, b] 구간 안의 말소리 구간을 조각 기준 상대 시간으로. 가까운 구간은 합침."""
    out = []
    for s, e in spans:
        s, e = max(s - pad, a), min(e + pad, b)
        if e <= s:
            continue
        s, e = s - a, e - a
        if out and s - out[-1][1] < join:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def _mlx_rows(model, audio, lang, **options):
    import mlx_whisper
    with _models_lock:  # mlx-whisper의 모델 캐시는 전역 하나라 동시에 부르지 않게
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language=lang, task="transcribe",
                                     verbose=None, condition_on_previous_text=False, **options)
    return [(seg["start"], seg["text"].strip()) for seg in res.get("segments", [])
            if seg["text"].strip()]


def run_mlx(audio, model, lang="ko", on_progress=None, start_at=0.0):
    """Apple Silicon GPU로 변환. 약 5분 조각씩 처리해서 진행률·이어하기가 되게 함.
    조용한 구간은 건너뜀(faster-whisper의 vad_filter와 같은 역할, 무음에서 헛말 방지).
    mlx-whisper의 clip_timestamps는 구간 사이를 실제로 건너뛰지 않아서 말소리 구간마다 따로 부름."""
    try:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        wav, total = _decode_from(audio, start_at)
        spans = _speech_spans(wav, SR)
        with _models_lock:
            if ("mlx", model) not in _models:
                log(f"모델 로드 중: {model} (Apple GPU, MLX)")
                ModelHolder.get_model(model, mx.float16)  # 내려받기·로드 실패를 여기서 잡아 CPU로 넘김
                _models[("mlx", model)] = True
    except Exception as e:  # noqa: BLE001
        raise _MLXStartError(f"{type(e).__name__}: {e}") from e

    cuts = _cut_points(wav, SR, spans=spans)
    for a, b in zip(cuts, cuts[1:]):
        clips = _clips_in(spans, a, b) if spans is not None else [[0.0, b - a]]
        for cs, ce in clips:
            piece = wav[int((a + cs) * SR):int((a + ce) * SR)]
            rows = _mlx_rows(model, piece, lang)
            retry = lambda clip: _mlx_rows(model, clip, lang, temperature=(0.2, 0.4, 0.6))  # noqa: E731
            for offset, text in _retry_loops(retry, piece, SR, rows):
                yield start_at + a + cs + offset, text
        if on_progress and total:
            on_progress(min((start_at + b) / total, 1.0))


def _run_whisper_mlx(audio, lang, model, gpu, on_progress, start_at):
    try:
        yield from run_mlx(audio, MLX_WHISPER_REPOS[model], lang, on_progress=on_progress,
                           start_at=start_at)
        return
    except _MLXStartError as exc:
        log(f"{model} Apple GPU(MLX) 시작 실패, CPU로 전환: {exc}")
    yield from run_whisper(audio, lang, model, gpu, on_progress, start_at)


COHERE_REPO = "CohereLabs/cohere-transcribe-03-2026"
COHERE_PIECE_S = 20      # Cohere는 타임스탬프를 안 줘서, 이 길이로 잘라 조각 시작 시각을 타임스탬프로 씀
COHERE_BATCH = 4


def _cohere_model():
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    with _models_lock:
        if "cohere" not in _models:
            log("모델 로드 중: Cohere Transcribe")
            _models["cohere"] = (
                AutoProcessor.from_pretrained(COHERE_REPO),
                CohereAsrForConditionalGeneration.from_pretrained(COHERE_REPO, device_map="auto"),
            )
        return _models["cohere"]


def _cohere_texts(processor, model, pieces, lang):
    """여러 조각을 한 번에 변환해 조각별 텍스트 목록을 돌려줌."""
    inputs = processor(pieces if len(pieces) > 1 else pieces[0], sampling_rate=SR,
                       return_tensors="pt", language=lang)
    # 조각이 모델 한도보다 길면 processor가 다시 잘라 주고, 이 인덱스로 이어 붙임
    chunk_index = inputs.pop("audio_chunk_index", None)
    inputs = inputs.to(model.device, dtype=model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=256)
    if chunk_index is not None:
        texts = processor.decode(outputs, skip_special_tokens=True,
                                 audio_chunk_index=chunk_index, language=lang)
    else:
        texts = processor.decode(outputs, skip_special_tokens=True)
    if isinstance(texts, str):
        texts = [texts]
    return [t.strip() for t in texts]


def _split_in_half(piece):
    """반복 인식이 난 조각을 조용한 틈(없으면 가운데)에서 둘로 나눔."""
    cuts = _cut_points(piece, SR, target_s=len(piece) / SR / 2)
    mid = cuts[1] if len(cuts) > 2 else len(piece) / SR / 2
    return piece[:int(mid * SR)], piece[int(mid * SR):]


def run_cohere(audio, lang="ko", on_progress=None, start_at=0.0, **_):
    processor, model = _cohere_model()
    wav, total = _decode_from(audio, start_at)
    cuts = _cut_points(wav, SR, target_s=COHERE_PIECE_S)
    spans = list(zip(cuts, cuts[1:]))
    for i in range(0, len(spans), COHERE_BATCH):
        batch = spans[i:i + COHERE_BATCH]
        pieces = [wav[int(a * SR):int(b * SR)] for a, b in batch]
        for (a, b), piece, text in zip(batch, pieces, _cohere_texts(processor, model, pieces, lang)):
            if _has_transcription_loop(text):
                # 반복 인식이 나면 조각을 반으로 나눠 다시 시도
                halves = [h for h in _split_in_half(piece) if len(h)]
                retry = " ".join(_cohere_texts(processor, model, halves, lang)) if halves else ""
                text = retry if retry and not _has_transcription_loop(retry) else LOOP_MARKER
            if text:
                yield start_at + a, text
        if on_progress and total:
            on_progress(min((start_at + batch[-1][1]) / total, 1.0))


def transcribe(audio, engine="ko", lang="ko", model="large-v3-turbo", gpu=None, on_progress=None,
               start_at=0.0):
    """엔진을 골라 (시작초, 텍스트)를 차례로 돌려줌. start_at초부터 시작할 수 있음(이어하기용)."""
    if engine == "ko":
        return run_ko(audio, gpu=gpu, on_progress=on_progress, start_at=start_at)
    if engine == "whisper":
        backend = gpu_backend(gpu)
        if backend == "mlx":
            if model in MLX_WHISPER_REPOS:
                return _run_whisper_mlx(audio, lang, model, gpu, on_progress, start_at)
            return run_whisper(audio, lang, model, gpu, on_progress, start_at)
        if model not in _WHISPER_OV_DIRS:
            backend = "other"
        if backend == "openvino":
            return _run_whisper_openvino(audio, lang, model, gpu, on_progress, start_at)
        if backend == "directml":
            return _run_whisper_onnx(audio, lang, model, gpu, on_progress, start_at)
        return run_whisper(audio, lang, model, gpu, on_progress, start_at)
    if engine == "cohere":
        return run_cohere(audio, lang, on_progress=on_progress, start_at=start_at)
    raise ValueError(f"알 수 없는 엔진: {engine} (가능: {', '.join(ENGINES)})")


DEFAULT_WHISPER_MODEL = "large-v3-turbo"


# ---------- 결과 저장 위치 ----------
# 결과는 원본 옆이 아니라 한 곳에 모음: <출력 폴더>/<녹음·노트 이름>/
# 출력 폴더는 환경 변수 TRANSCRIBE_OUTPUT_DIR > 설치 폴더의 config.json {"output_dir": ...} > <설치 폴더>/output
def output_root() -> Path:
    env = os.environ.get("TRANSCRIBE_OUTPUT_DIR")
    if env:
        return Path(env).expanduser()
    cfg = HERE / "config.json"
    if cfg.is_file():
        try:
            import json
            value = json.loads(cfg.read_text(encoding="utf-8")).get("output_dir")
            if value:
                return Path(value).expanduser()
        except (ValueError, OSError) as exc:
            log(f"config.json을 읽지 못해 기본 출력 폴더를 써요: {exc}")
    return HERE / "output"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def work_dir(source, create: bool = False) -> Path:
    """녹음·노트 하나의 결과를 모으는 폴더. 이미 출력 폴더 안에 있는 파일(노트에서 꺼낸 녹음 등)은
    그 파일이 있는 폴더를 그대로 씀. 이름이 같은 다른 파일과는 source.txt로 구분해 겹치지 않게 함."""
    src = Path(source).resolve()
    root = output_root()
    if _is_within(src, root):
        return src.parent
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", src.stem).strip(" .") or "recording"
    d = root / name
    marker = d / "source.txt"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() != str(src):
        import hashlib
        d = root / f"{name}_{hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:6]}"
        marker = d / "source.txt"
    if create and not marker.is_file():
        d.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(src), encoding="utf-8")
    return d


def _result_suffix(engine, model) -> str:
    """whisper는 모델마다 결과가 달라서 기본 모델(turbo)이 아니면 모델 이름을 붙임."""
    if engine == "whisper" and model not in (DEFAULT_WHISPER_MODEL, "turbo"):
        safe = re.sub(r"[^\w.\-]+", "_", Path(str(model)).name) or "custom"
        return f".whisper.{safe}.txt"
    return f".{engine}.txt"


def output_path(audio, engine, model=DEFAULT_WHISPER_MODEL) -> Path:
    """결과 파일 경로. 예: output/강의/강의.ko.txt, 강의.whisper.txt(turbo), 강의.whisper.large-v3.txt"""
    return work_dir(audio) / (Path(audio).stem + _result_suffix(engine, model))


def _legacy_output_path(audio, engine, model) -> Path:
    """예전 버전처럼 원본 옆에 저장된 결과."""
    return Path(audio).with_suffix(_result_suffix(engine, model))


def finished_outputs(audio) -> list[str]:
    """이 녹음에 대해 이미 끝난 변환 결과 이름들 (예: ['ko', 'whisper', 'whisper.large-v3'])."""
    audio = Path(audio)
    prefix = audio.stem + "."
    found = set()
    for folder in {work_dir(audio), audio.parent}:   # 새 위치 + 예전 위치(원본 옆)
        for p in folder.glob(f"{glob.escape(audio.stem)}.*.txt"):
            name = p.name[len(prefix):-len(".txt")]
            if name.split(".")[0] in ENGINES:
                found.add(name)
    return sorted(found)


def format_line(start, text) -> str:
    return f"[{fmt(start)}] {text}" if start is not None else text


_TS = re.compile(r"^\[(\d+):(\d\d):(\d\d)\] ")


def _resume_point(part: Path):
    """중단된 .part 파일에서 이어 시작할 시점(초). 마지막 줄은 덜 써졌을 수 있어 버리고 그 줄 시작부터 다시."""
    if not part.exists():
        return 0.0
    lines = part.read_text(encoding="utf-8", errors="replace").splitlines()
    stamps = [(i, _TS.match(l)) for i, l in enumerate(lines)]
    stamps = [(i, m) for i, m in stamps if m]
    if not stamps:
        part.unlink()
        return 0.0
    i, m = stamps[-1]
    h, mi, s = map(int, m.groups())
    part.write_text("".join(l + "\n" for l in lines[:i]), encoding="utf-8")
    return float(h * 3600 + mi * 60 + s)


def transcribe_to_file(audio, engine="ko", lang="ko", model="large-v3-turbo", gpu=None,
                       on_progress=None, echo=False, *, force=False) -> Path:
    """녹음을 <출력 폴더>/<이름>/<이름>.<engine>.txt로 변환. 중간에 끊겨도 다음 실행 때 끊긴 지점부터 이어서 함.
    변환 중에는 <이름>.<engine>.txt.part에 한 줄씩 바로 저장하고, 끝까지 성공하면 이름을 바꿈."""
    work_dir(audio, create=True)
    out = output_path(audio, engine, model)
    if out.exists() and not force:
        return out
    legacy = _legacy_output_path(audio, engine, model)
    if legacy.is_file() and not force and legacy != out:
        # 예전 버전이 원본 옆에 만든 결과는 새 위치로 복사해서 그대로 씀
        import shutil
        shutil.copy2(legacy, out)
        return out
    part = out.with_suffix(out.suffix + ".part")
    start_at = _resume_point(part) if not force else 0.0
    if start_at:
        log(f"이어서 변환: {fmt(start_at)}부터")
    with part.open("a" if start_at else "w", encoding="utf-8") as f:
        for start, text in transcribe(audio, engine, lang, model, gpu, on_progress, start_at):
            line = format_line(start, text)
            if echo:
                print(line, flush=True)
            f.write(line + "\n")
            f.flush()  # 강제 종료돼도 여기까지는 남음
    part.replace(out)
    return out


# ---------- 자가 진단 ----------
def _version(mod):
    try:
        import importlib.metadata as md
        return md.version(mod)
    except Exception:  # noqa: BLE001
        return None


def _dir_size_gb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9


def doctor(run_test: bool = True) -> str:
    """설치·장치 상태를 점검하고, 고른 가속 방식으로 실제로 짧은 변환을 돌려 확인."""
    import platform
    import tempfile
    import time
    import wave

    lines = [f"Python {platform.python_version()} · {platform.system()} {platform.release()} ({platform.machine()})"]
    pkgs = ["faster-whisper", "ctranslate2", "av", "mlx", "mlx-whisper", "openvino", "openvino-genai", "onnxruntime",
            "onnxruntime-directml", "optimum", "optimum-intel", "optimum-onnx", "transformers", "torch"]
    lines.append("패키지: " + ", ".join(f"{p} {v}" for p in pkgs if (v := _version(p))))

    cuda = cuda_available()
    ov = ov_device()
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        providers = []
    lines += [
        f"Apple Silicon(MLX): {'사용 가능' if mlx_available() else '없음'}",
        f"NVIDIA CUDA: {'있음' if cuda else '없음'}",
        f"인텔 GPU(OpenVINO): {ov or '없음'}",
        f"ONNX Runtime 실행 장치: {', '.join(providers) or '없음'}"
        + (" → DirectML 사용 가능" if directml_available() else ""),
    ]
    backend = gpu_backend(None)
    lines.append(f"자동 선택: {backend}")
    lines.append(f"결과 저장 폴더: {output_root()}")

    models = {"ko (CPU·CUDA)": KO_DIR, "ko (Apple GPU)": KO_MLX_DIR, "ko (인텔 GPU)": KO_OV_DIR, "ko (DirectML)": KO_ONNX_DIR,
              "large-v3 (인텔 GPU)": LARGE_V3_OV_DIR, "large-v3-turbo (인텔 GPU)": LARGE_V3_TURBO_OV_DIR,
              "large-v3 (DirectML)": _WHISPER_ONNX_DIRS["large-v3"],
              "large-v3-turbo (DirectML)": _WHISPER_ONNX_DIRS["large-v3-turbo"]}
    ready = [name for name, d in models.items() if d.is_dir() and any(d.iterdir())]
    lines.append("준비된 모델: " + (", ".join(ready) or "없음"))
    stale = [d for d in (HERE / "models").glob("tmp*") if d.is_dir()] if (HERE / "models").is_dir() else []
    if stale:
        lines.append(f"중단된 변환 임시 폴더 {len(stale)}개 ({sum(map(_dir_size_gb, stale)):.1f}GB) — 다음 변환 때 3시간 지난 것은 자동 정리")

    if not run_test:
        return "\n".join(lines)

    need = {"mlx": KO_MLX_DIR, "cuda": KO_DIR, "openvino": KO_OV_DIR, "directml": KO_ONNX_DIR, "cpu": KO_DIR}[backend]
    if not need.is_dir():
        lines.append(f"실행 확인: 건너뜀 — {backend}용 한국어 모델이 아직 준비되지 않았어요 (install.py 또는 setup.bat 실행)")
        return "\n".join(lines)

    # 10초짜리 시험 음성(조용한 구간 + 짧은 소리)으로 실제 경로를 한 번 돌려 봄
    import numpy as np

    t = np.arange(10 * SR) / SR
    sig = (0.2 * np.sin(2 * np.pi * 220 * t) * ((t % 2) < 1)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "check.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((sig * 32767).astype("<i2").tobytes())
        start = time.time()
        try:
            if backend == "mlx":
                list(run_mlx(str(wav_path), str(KO_MLX_DIR), "ko"))
                where = "MLX (Apple GPU)"
            elif backend == "openvino":
                list(run_openvino(str(wav_path), str(KO_OV_DIR), "ko"))
                where = f"OpenVINO {ov}"
            elif backend == "directml":
                list(run_onnx(str(wav_path), str(KO_ONNX_DIR), "ko"))
                pipe = _models[("onnx", str(KO_ONNX_DIR), _ort_provider())]
                where = "ONNX Runtime " + ", ".join(getattr(pipe.model, "providers", []) or [_ort_provider()])
            else:
                list(run_whisper(str(wav_path), "ko", str(KO_DIR), backend == "cuda"))
                m = next(v for k, v in _models.items() if k[0] == "fw")
                where = f"faster-whisper {m.model.device}"
            lines.append(f"실행 확인: 성공 — {where}에서 10초 시험 음성을 {time.time() - start:.1f}초에 처리")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"실행 확인: 실패 ({backend}) — {type(exc).__name__}: {exc}\n"
                         "  → 실제 변환에서는 CPU로 자동 전환돼요. 드라이버·설치 옵션을 확인하세요.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="?", help="mp3, m4a, wav, mp4 등")
    p.add_argument("--engine", choices=ENGINES, default="ko")
    p.add_argument("--model", default="large-v3-turbo",
                   help="whisper 엔진 전용: tiny/base/small/medium/large-v3/large-v3-turbo")
    p.add_argument("--lang", default="ko")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", dest="gpu", action="store_const", const=True,
                   help="GPU 사용 (Apple Silicon MLX, NVIDIA CUDA, 지원 모델은 Intel OpenVINO·DirectML)")
    g.add_argument("--cpu", dest="gpu", action="store_const", const=False, help="CPU 강제 사용")
    p.add_argument("--force", action="store_true", help="이미 변환된 결과가 있어도 처음부터 다시")
    p.add_argument("--choose-model", action="store_true", help="녹음과 음성 인식 모델을 직접 선택")
    p.add_argument("--doctor", action="store_true", help="설치·GPU 상태 점검 (짧은 시험 변환 포함)")
    p.set_defaults(gpu=None)
    a = p.parse_args()

    if a.doctor:
        print(doctor(run_test=True))
        return
    if a.choose_model:
        if a.audio is None:
            a.audio = input("녹음 파일 경로를 붙여넣으세요: ").strip().strip('"')
        choices = {
            "1": ("ko", "large-v3-turbo"),
            "2": ("whisper", "large-v3"),
            "3": ("whisper", "large-v3-turbo"),
        }
        print("1. 한국어 특화 BatiSay Turbo\n2. Whisper large-v3 (Intel GPU 지원)\n3. Whisper large-v3-turbo (Intel GPU 지원)")
        choice = input("모델 번호 [1]: ").strip() or "1"
        if choice not in choices:
            p.error("모델 번호는 1, 2, 3 중 하나여야 합니다.")
        a.engine, a.model = choices[choice]
    if not a.audio or not Path(a.audio).is_file():
        p.error("존재하는 녹음 파일 경로를 입력하세요.")
    out = output_path(a.audio, a.engine, a.model)
    if out.exists() and not a.force:
        print(f"이미 변환돼 있어요: {out}  (다시 하려면 --force)")
        return
    out = transcribe_to_file(a.audio, a.engine, a.lang, a.model, a.gpu, echo=True, force=a.force)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
