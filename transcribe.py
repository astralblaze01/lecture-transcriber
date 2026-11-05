"""
녹음 파일 → 텍스트 변환 (로컬 실행, 엔진 3종 선택)
명령줄에서 직접 써도 되고, transcribe_mcp.py가 이 파일을 불러서 Claude에 연결해요.

엔진
  ko      : batiai/batisay-ko-turbo (Whisper turbo 한국어 파인튜닝, 타임스탬프 O) ← 한국어 추천
  whisper : faster-whisper 기본 모델 (다국어, 타임스탬프 O)
  cohere  : CohereLabs/cohere-transcribe-03-2026 (오픈 ASR 리더보드 1위, 타임스탬프 X, GPU 권장)

사용
  python transcribe.py 녹음.m4a                       # ko 엔진
  python transcribe.py 녹음.m4a --engine whisper --model large-v3-turbo --lang en
  python transcribe.py 녹음.m4a --engine cohere
  (GPU 자동 사용: NVIDIA → CUDA, 인텔 내장/Arc → OpenVINO(ko 엔진), 없으면 CPU. --cpu로 끌 수 있음)
결과
  녹음.<engine>.txt 가 같은 폴더에 생성

cohere 엔진은 추가 설치가 필요해요:
  pip install "transformers>=5.4.0" accelerate librosa soundfile sentencepiece protobuf
  + https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 에서 약관 동의 후 `hf auth login`
"""
import argparse
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
            "한국어 모델 변환에 필요한 패키지가 없어요. setup.bat을 다시 실행하거나:\n"
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
    total = None
    if start_at:
        audio, total = _decode_from(audio, start_at)
    segments, info = m.transcribe(audio, language=lang, vad_filter=True)
    total = total or info.duration or 0
    for seg in segments:
        if on_progress and total:
            on_progress(min((start_at + seg.end) / total, 1.0))
        yield start_at + seg.start, seg.text.strip()


def run_ko(audio, gpu=None, on_progress=None, start_at=0.0, **_):
    # 자동 모드: NVIDIA(CUDA) → 인텔 GPU(OpenVINO) → CPU 순서
    if gpu is not False and not cuda_available() and intel_gpu_available():
        try:
            yield from run_openvino(audio, ensure_ko_ov_model(), "ko", on_progress=on_progress,
                                    start_at=start_at)
            return
        except _OVStartError as e:
            log(f"인텔 GPU로 시작하지 못해 CPU로 바꿔요: {e}")
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


class _OVStartError(RuntimeError):
    """OpenVINO로 시작조차 못 했을 때(설치·변환·장치 문제). 이미 텍스트를 내보낸 뒤의 오류와 구분."""


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

def _cut_points(wav, sr, target_s=300):
    """긴 녹음을 약 target_s초 단위로 자르되, 말하는 중간이 아니라 조용한 틈에서 자름."""
    total = len(wav) / sr
    if total <= target_s * 1.2:
        return [0.0, total]
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        speech = get_speech_timestamps(wav, VadOptions(min_silence_duration_ms=300))
        gaps = [(a["end"] + b["start"]) / 2 / sr for a, b in zip(speech, speech[1:])]
    except Exception:  # noqa: BLE001
        gaps = []
    cuts, t = [0.0], target_s
    while t < total - target_s * 0.2:
        near = [g for g in gaps if abs(g - t) < 30 and g > cuts[-1] + 30]
        c = min(near, key=lambda g: abs(g - t)) if near else t
        cuts.append(c)
        t = c + target_s
    cuts.append(total)
    return cuts


_REPEATED_PHRASE = re.compile(
    r"(?<!\S)(\S+(?:\s+\S+){0,5}?)(?:\s+\1){5,}(?=\s|$)"
)


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
    for index, (offset, text) in enumerate(rows):
        if not text:
            continue
        if _has_transcription_loop(text):
            next_offset = rows[index + 1][0] if index + 1 < len(rows) else len(piece) / sr
            end = min(len(piece) / sr, max(offset + 2.0, next_offset), offset + 25.0)
            clip = piece[int(offset * sr):int(end * sr)]
            if len(clip):
                try:
                    text = " ".join(retry for _, retry in _openvino_rows(pipe, clip, lang)).strip()
                except RuntimeError as exc:
                    log(f"반복 인식 구간 재시도 실패: {exc}")
                    text = ""
            if not len(clip) or not text or _has_transcription_loop(text):
                text = "[반복 인식 오류 · 재변환으로 복구하지 못함]"
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


def run_cohere(audio, lang="ko", on_progress=None, **_):
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    from transformers.audio_utils import load_audio

    repo = "CohereLabs/cohere-transcribe-03-2026"
    with _models_lock:
        if "cohere" not in _models:
            log("모델 로드 중: Cohere Transcribe")
            _models["cohere"] = (
                AutoProcessor.from_pretrained(repo),
                CohereAsrForConditionalGeneration.from_pretrained(repo, device_map="auto"),
            )
    processor, model = _models["cohere"]

    wav = load_audio(audio, sampling_rate=16000)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", language=lang)
    # 긴 녹음은 processor가 자동으로 잘라주고, 이 인덱스로 다시 이어붙임
    chunk_index = inputs.pop("audio_chunk_index", None)
    inputs = inputs.to(model.device, dtype=model.dtype)

    outputs = model.generate(**inputs, max_new_tokens=256)
    if chunk_index is not None:
        text = processor.decode(
            outputs, skip_special_tokens=True,
            audio_chunk_index=chunk_index, language=lang,
        )[0]
    else:
        text = processor.decode(outputs, skip_special_tokens=True)
        if isinstance(text, list):
            text = text[0]
    if on_progress:
        on_progress(1.0)
    # 타임스탬프가 없으니 문장 단위로만 줄바꿈
    for sent in text.replace(". ", ".\n").replace("? ", "?\n").splitlines():
        if sent.strip():
            yield None, sent.strip()


def transcribe(audio, engine="ko", lang="ko", model="large-v3-turbo", gpu=None, on_progress=None,
               start_at=0.0):
    """엔진을 골라 (시작초, 텍스트)를 차례로 돌려줌. start_at초부터 시작할 수 있음(이어하기용)."""
    if engine == "ko":
        return run_ko(audio, gpu=gpu, on_progress=on_progress, start_at=start_at)
    if engine == "whisper":
        if model in _WHISPER_OV_DIRS and gpu is not False and not cuda_available() and intel_gpu_available():
            return _run_whisper_openvino(audio, lang, model, gpu, on_progress, start_at)
        return run_whisper(audio, lang, model, gpu, on_progress, start_at)
    if engine == "cohere":
        return run_cohere(audio, lang, on_progress=on_progress)
    raise ValueError(f"알 수 없는 엔진: {engine} (가능: {', '.join(ENGINES)})")


def output_path(audio, engine, model="large-v3-turbo") -> Path:
    suffix = ".whisper.large-v3.txt" if engine == "whisper" and model == "large-v3" else f".{engine}.txt"
    return Path(audio).with_suffix(suffix)


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
    """녹음을 <이름>.<engine>.txt로 변환. 중간에 끊겨도 다음 실행 때 끊긴 지점부터 이어서 함.
    변환 중에는 <이름>.<engine>.txt.part에 한 줄씩 바로 저장하고, 끝까지 성공하면 이름을 바꿈."""
    out = output_path(audio, engine, model)
    if out.exists() and not force:
        return out
    part = out.with_suffix(out.suffix + ".part")
    start_at = _resume_point(part) if not force and engine != "cohere" else 0.0  # cohere는 타임스탬프가 없어 처음부터
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="?", help="mp3, m4a, wav, mp4 등")
    p.add_argument("--engine", choices=ENGINES, default="ko")
    p.add_argument("--model", default="large-v3-turbo",
                   help="whisper 엔진 전용: tiny/base/small/medium/large-v3/large-v3-turbo")
    p.add_argument("--lang", default="ko")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", dest="gpu", action="store_const", const=True,
                   help="GPU 사용 (NVIDIA CUDA, 지원 모델은 Intel OpenVINO)")
    g.add_argument("--cpu", dest="gpu", action="store_const", const=False, help="CPU 강제 사용")
    p.add_argument("--force", action="store_true", help="이미 변환된 결과가 있어도 처음부터 다시")
    p.add_argument("--choose-model", action="store_true", help="녹음과 음성 인식 모델을 직접 선택")
    p.set_defaults(gpu=None)
    a = p.parse_args()

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
