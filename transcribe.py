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
결과
  녹음.<engine>.txt 가 같은 폴더에 생성

cohere 엔진은 추가 설치가 필요해요:
  pip install "transformers>=5.4.0" accelerate librosa soundfile sentencepiece protobuf
  + https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 에서 약관 동의 후 `hf auth login`
"""
import argparse
import os
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


def _whisper_model(name_or_path, gpu):
    from faster_whisper import WhisperModel

    key = ("fw", name_or_path, gpu)
    with _models_lock:
        if key not in _models:
            device, ctype = ("cuda", "float16") if gpu else ("cpu", "int8")
            log(f"모델 로드 중: {name_or_path} ({device})")
            _models[key] = WhisperModel(
                name_or_path, device=device, compute_type=ctype,
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

def run_whisper(audio, lang="ko", model="large-v3-turbo", gpu=False, on_progress=None):
    m = _whisper_model(model, gpu)
    segments, info = m.transcribe(audio, language=lang, vad_filter=True)
    total = info.duration or 0
    for seg in segments:
        if on_progress and total:
            on_progress(min(seg.end / total, 1.0))
        yield seg.start, seg.text.strip()


def run_ko(audio, gpu=False, on_progress=None, **_):
    yield from run_whisper(audio, "ko", ensure_ko_model(), gpu, on_progress)


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


def transcribe(audio, engine="ko", lang="ko", model="large-v3-turbo", gpu=False, on_progress=None):
    """엔진을 골라 (시작초, 텍스트)를 차례로 돌려줌."""
    if engine == "ko":
        return run_ko(audio, gpu=gpu, on_progress=on_progress)
    if engine == "whisper":
        return run_whisper(audio, lang, model, gpu, on_progress)
    if engine == "cohere":
        return run_cohere(audio, lang, on_progress=on_progress)
    raise ValueError(f"알 수 없는 엔진: {engine} (가능: {', '.join(ENGINES)})")


def output_path(audio, engine) -> Path:
    return Path(audio).with_suffix(f".{engine}.txt")


def format_line(start, text) -> str:
    return f"[{fmt(start)}] {text}" if start is not None else text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="mp3, m4a, wav, mp4 등")
    p.add_argument("--engine", choices=ENGINES, default="ko")
    p.add_argument("--model", default="large-v3-turbo",
                   help="whisper 엔진 전용: tiny/base/small/medium/large-v3-turbo")
    p.add_argument("--lang", default="ko")
    p.add_argument("--gpu", action="store_true", help="NVIDIA GPU가 있으면 사용")
    a = p.parse_args()

    out = output_path(a.audio, a.engine)
    with out.open("w", encoding="utf-8") as f:
        for start, text in transcribe(a.audio, a.engine, a.lang, a.model, a.gpu):
            line = format_line(start, text)
            print(line)
            f.write(line + "\n")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
