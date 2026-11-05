"""모델 없이 돌아가는 핵심 동작 테스트: pytest -q"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lecture_tools as lt  # noqa: E402
import transcribe as tr  # noqa: E402


@pytest.fixture(autouse=True)
def _output_dir(tmp_path, monkeypatch):
    """테스트마다 결과를 임시 출력 폴더에 저장"""
    out = tmp_path / "output"
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_DIR", str(out))
    return out


# ---------- 결과 저장 위치 ----------
def test_results_go_to_fixed_output_folder(tmp_path, _output_dir, monkeypatch):
    audio = tmp_path / "Downloads" / "강의.m4a"
    audio.parent.mkdir()
    audio.write_bytes(b"x")
    monkeypatch.setattr(tr, "transcribe", lambda *a, **k: iter([(0.0, "문장")]))
    out = tr.transcribe_to_file(str(audio))
    assert out == _output_dir / "강의" / "강의.ko.txt"
    assert (_output_dir / "강의" / "source.txt").read_text(encoding="utf-8") == str(audio.resolve())
    assert not list(audio.parent.glob("*.txt"))          # 원본 폴더는 건드리지 않음
    assert tr.finished_outputs(audio) == ["ko"]


def test_same_name_from_different_folders_does_not_collide(tmp_path, _output_dir):
    a, b = tmp_path / "a" / "강의.m4a", tmp_path / "b" / "강의.m4a"
    for f in (a, b):
        f.parent.mkdir()
        f.write_bytes(b"x")
    first = tr.work_dir(a, create=True)
    second = tr.work_dir(b, create=True)
    assert first == _output_dir / "강의" and second != first and second.name.startswith("강의_")
    assert tr.work_dir(a) == first and tr.work_dir(b) == second   # 다시 불러도 같은 폴더


def test_files_already_in_output_folder_keep_their_folder(_output_dir):
    extracted = _output_dir / "Lecture12" / "추출" / "01_rec.m4a"
    extracted.parent.mkdir(parents=True)
    extracted.write_bytes(b"x")
    assert tr.output_path(extracted, "ko") == extracted.parent / "01_rec.ko.txt"


def test_legacy_result_next_to_source_is_reused(tmp_path, _output_dir, monkeypatch):
    audio = tmp_path / "rec.m4a"
    audio.write_bytes(b"x")
    (tmp_path / "rec.ko.txt").write_text("[00:00:00] 예전 결과\n", encoding="utf-8")
    monkeypatch.setattr(tr, "transcribe", lambda *a, **k: pytest.fail("다시 변환하면 안 됨"))
    out = tr.transcribe_to_file(str(audio))
    assert out.parent == _output_dir / "rec" and "예전 결과" in out.read_text(encoding="utf-8")


def test_output_dir_from_config_json(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_OUTPUT_DIR")
    monkeypatch.setattr(tr, "HERE", tmp_path)
    assert tr.output_root() == tmp_path / "output"
    (tmp_path / "config.json").write_text('{"output_dir": "D:/강의결과"}', encoding="utf-8")
    assert tr.output_root() == Path("D:/강의결과")


# ---------- 결과 파일 이름 ----------
@pytest.mark.parametrize("engine,model,name", [
    ("ko", "large-v3-turbo", "강의.ko.txt"),
    ("whisper", "large-v3-turbo", "강의.whisper.txt"),
    ("whisper", "turbo", "강의.whisper.txt"),
    ("whisper", "large-v3", "강의.whisper.large-v3.txt"),
    ("whisper", "small", "강의.whisper.small.txt"),
])
def test_output_path_is_unique_per_model(engine, model, name):
    assert tr.output_path("강의.m4a", engine, model).name == name


def test_finished_outputs_lists_every_engine_and_model(tmp_path):
    audio = tmp_path / "강의.m4a"
    for suffix in ("ko", "whisper", "whisper.large-v3"):
        (tmp_path / f"강의.{suffix}.txt").write_text("x", encoding="utf-8")
    (tmp_path / "강의.notes.txt").write_text("x", encoding="utf-8")  # 변환 결과가 아님
    assert tr.finished_outputs(audio) == ["ko", "whisper", "whisper.large-v3"]


# ---------- 반복 인식 감지·복구 ----------
@pytest.mark.parametrize("text,loop", [
    ("네 그래서 컨볼루션을 하면 공간 정보가 유지됩니다", False),
    ("그래서 " + "이렇게 들어가는 거죠 " * 8, True),
    ("0 1 1 1 1 1 0 " + "1 " * 10, False),  # 비트 스터핑 같은 짧은 숫자열은 정상
])
def test_loop_detection(text, loop):
    assert tr._has_transcription_loop(text) is loop


class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeWhisper:
    """첫 호출은 반복 구간을 포함해 돌려주고, 재시도 호출에는 retry_text를 돌려줌."""
    def __init__(self, retry_text):
        self.retry_text, self.calls = retry_text, []

    def transcribe(self, audio, **kw):
        self.calls.append((len(audio) / tr.SR, kw))
        if len(self.calls) == 1:
            return iter([_Seg(0, 4, "정상 문장"), _Seg(4, 9, "이렇게 들어가는 거죠 " * 8)]), None
        return iter([_Seg(0, 5, self.retry_text)]), None


@pytest.mark.parametrize("retry_text,expected", [
    ("다시 받아쓴 정상 문장", "다시 받아쓴 정상 문장"),
    ("이렇게 들어가는 거죠 " * 8, tr.LOOP_MARKER),
])
def test_faster_whisper_loop_is_retried_without_context(monkeypatch, retry_text, expected):
    fake = _FakeWhisper(retry_text)
    monkeypatch.setattr(tr, "_whisper_model", lambda *a, **k: fake)
    monkeypatch.setattr(tr, "_decode_from", lambda audio, s: (np.zeros(tr.SR * 20, np.float32), 20.0))
    rows = list(tr.run_whisper("a.m4a", "ko", "m"))
    assert rows == [(0.0, "정상 문장"), (4.0, expected)]
    clip_seconds, kw = fake.calls[1]
    assert clip_seconds == 5 and kw["condition_on_previous_text"] is False


# ---------- 이어하기 ----------
def test_resume_from_interrupted_part(tmp_path, monkeypatch):
    audio = tmp_path / "rec.m4a"
    audio.write_bytes(b"x")
    starts = []

    def engine(audio, engine="ko", lang="ko", model="m", gpu=None, on_progress=None, start_at=0.0):
        starts.append(start_at)
        for t in range(0, 600, 60):
            if t < start_at:
                continue
            if t == 300 and len(starts) == 1:
                raise KeyboardInterrupt  # 변환 도중 강제 종료
            yield float(t), f"문장 {t}"

    monkeypatch.setattr(tr, "transcribe", engine)
    with pytest.raises(KeyboardInterrupt):
        tr.transcribe_to_file(str(audio))
    out = tr.transcribe_to_file(str(audio))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert starts == [0.0, 240.0]           # 마지막(덜 써졌을 수 있는) 줄부터 다시
    assert len(lines) == 10 and len(set(lines)) == 10  # 빠진 줄도 겹친 줄도 없음
    assert not out.with_suffix(".txt.part").exists()


def test_force_ignores_existing_result_and_part(tmp_path, monkeypatch):
    audio = tmp_path / "rec.m4a"
    audio.write_bytes(b"x")
    wd = tr.work_dir(audio, create=True)
    (wd / "rec.ko.txt").write_text("옛 결과\n", encoding="utf-8")
    (wd / "rec.ko.txt.part").write_text("[00:05:00] 옛 조각\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(tr, "transcribe", lambda *a, start_at=0.0, **k: seen.append(start_at) or iter([(0.0, "새 결과")]))
    out = tr.transcribe_to_file(str(audio), force=True)
    assert seen == [0.0] and out.read_text(encoding="utf-8") == "[00:00:00] 새 결과\n"


# ---------- 긴 녹음 자르기 ----------
def test_cut_points_land_in_silence(monkeypatch):
    import faster_whisper.vad as vad
    total = 13 * 60
    speech = [{"start": t * tr.SR, "end": (t + 18) * tr.SR} for t in range(0, total, 20)]
    monkeypatch.setattr(vad, "get_speech_timestamps", lambda w, o=None, **k: speech)
    cuts = tr._cut_points(np.zeros(total * tr.SR, np.float32), tr.SR)
    assert cuts[0] == 0 and cuts[-1] == total
    for c in cuts[1:-1]:
        assert c % 20 == 19  # 말 사이(18~20초 구간)에서 자름


# ---------- 삼성노트 ----------
def test_extract_note_by_signature_and_safely(tmp_path):
    note = tmp_path / "lec.sdocx"
    with zipfile.ZipFile(note, "w") as z:
        z.writestr("media/a.dat", b"\x00\x00\x00\x18ftyp3gp4" + b"\x00" * 100)
        z.writestr("media/b", b"%PDF-1.7" + b"\x00" * 100)
        z.writestr("../evil.txt", b"plain text " * 20)
    info = lt.extract_note(note)
    assert [p.suffix for p in info["audio"]] == [".m4a"]
    assert [p.suffix for p in info["pdf"]] == [".pdf"]
    assert all(p.parent == info["dir"] for p in info["audio"] + info["pdf"])
    assert not (tmp_path / "evil.txt").exists()


# ---------- 책 HTML ----------
def test_book_html_math_lists_and_exam_box():
    md = ("# 제목\n\n문단 바로 다음 목록:\n- 하나\n- 둘\n\n"
          "인라인 $W_{1} x$ 와 블록\n\n$$H' = (H - K + 2P)/S + 1$$\n\n"
          "> **시험 포인트** ==형광펜== 내용\n\n> **계산 확인** 메모\n\n`a == b` 코드는 그대로\n")
    html = lt.markdown_to_book_html(md, "제목")
    assert "<ul>" in html and "<li>하나</li>" in html
    assert '<span class="math inline">W_{1} x</span>' in html
    assert '<div class="math display">' in html and "<p><div" not in html
    assert '<blockquote class="exam">' in html and html.count("<blockquote>") == 1
    assert "<mark>형광펜</mark>" in html and "<code>a == b</code>" in html
    assert "KEEPTOKEN" not in html


# ---------- Cohere: 조각별 타임스탬프·이어하기·반복 복구 ----------
class _CohereInputs(dict):
    def to(self, *a, **k):
        return self


class _FakeCohere:
    """조각 길이(초)를 텍스트로 돌려주고, loop_at 길이의 조각에는 반복 텍스트를 돌려줌."""
    device, dtype = "cpu", None

    def __init__(self, loop_seconds=()):
        self.loop_seconds, self.batches = set(loop_seconds), []

    def __call__(self, audio, sampling_rate, return_tensors, language):
        pieces = audio if isinstance(audio, list) else [audio]
        self.batches.append([round(len(p) / tr.SR) for p in pieces])
        return _CohereInputs(lengths=[len(p) / tr.SR for p in pieces])

    def generate(self, lengths, max_new_tokens):
        return lengths

    def decode(self, lengths, skip_special_tokens):
        return [("반복 문장입니다 " * 8) if round(n) in self.loop_seconds else f"{round(n)}초 분량" for n in lengths]


def _setup_cohere(monkeypatch, fake, total=100):
    import faster_whisper.vad as vad
    monkeypatch.setattr(tr, "_cohere_model", lambda: (fake, fake))
    wav = np.zeros(total * tr.SR, np.float32)
    monkeypatch.setattr(tr, "_decode_from", lambda audio, s: (wav[int(s * tr.SR):], total))
    # 10초마다 말이 끊기는 녹음 (9~10초 구간이 조용함)
    monkeypatch.setattr(vad, "get_speech_timestamps",
                        lambda w, o=None, **k: [{"start": t * tr.SR, "end": (t + 9) * tr.SR}
                                                for t in range(0, int(len(w) / tr.SR), 10)])


def test_cohere_emits_piece_timestamps_in_batches(monkeypatch):
    fake = _FakeCohere()
    _setup_cohere(monkeypatch, fake)
    rows = list(tr.run_cohere("a.m4a"))
    starts = [r[0] for r in rows]
    assert starts[0] == 0 and starts == sorted(starts) and len(rows) >= 4
    assert all(s % 10 == 9.5 for s in starts[1:])        # 조용한 틈에서 자름
    assert all(len(b) <= tr.COHERE_BATCH for b in fake.batches)


def test_cohere_resumes_with_absolute_timestamps(monkeypatch):
    fake = _FakeCohere()
    _setup_cohere(monkeypatch, fake)
    rows = list(tr.run_cohere("a.m4a", start_at=40.0))
    assert rows[0][0] == 40.0 and all(r[0] >= 40.0 for r in rows)


def test_cohere_loop_piece_is_split_and_retried(monkeypatch):
    fake = _FakeCohere()
    _setup_cohere(monkeypatch, fake, total=20)          # 20초 → 조각 하나
    fake.loop_seconds = {20}
    rows = list(tr.run_cohere("a.m4a"))
    assert fake.batches[-1] == [10, 10] or sum(fake.batches[-1]) == 20   # 반으로 나눠 재시도
    assert len(rows) == 1 and not tr._has_transcription_loop(rows[0][1])
    fake.batches.clear()
    fake.loop_seconds = {20, 10}                         # 나눠도 반복이면 표시
    assert list(tr.run_cohere("a.m4a")) == [(0.0, tr.LOOP_MARKER)]


# ---------- GPU 선택·DirectML·자가 진단 ----------
@pytest.mark.parametrize("mlx,cuda,intel,dml,gpu,expected", [
    (True, False, False, False, None, "mlx"),
    (False, True, True, True, None, "cuda"),
    (False, False, True, True, None, "openvino"),
    (False, False, False, True, None, "directml"),
    (False, False, False, False, None, "cpu"),
    (True, True, True, True, False, "cpu"),          # --cpu 강제
])
def test_gpu_backend_order(monkeypatch, mlx, cuda, intel, dml, gpu, expected):
    monkeypatch.setattr(tr, "mlx_available", lambda: mlx)
    monkeypatch.setattr(tr, "cuda_available", lambda: cuda)
    monkeypatch.setattr(tr, "intel_gpu_available", lambda: intel)
    monkeypatch.setattr(tr, "directml_available", lambda: dml)
    assert tr.gpu_backend(gpu) == expected


def test_gpu_start_failure_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(tr, "gpu_backend", lambda gpu=None: "directml")
    monkeypatch.setattr(tr, "ensure_onnx_model", lambda *a: (_ for _ in ()).throw(tr._GPUStartError("no driver")))
    monkeypatch.setattr(tr, "ensure_ko_model", lambda: "ct2")
    monkeypatch.setattr(tr, "run_whisper", lambda *a, **k: iter([(0.0, "CPU로 변환")]))
    assert list(tr.run_ko("a.m4a")) == [(0.0, "CPU로 변환")]


def test_onnx_rows_reads_pipeline_chunks():
    pipe = lambda audio, **kw: {"text": "가 나", "chunks": [{"timestamp": (0.0, 2.0), "text": " 가"},
                                                         {"timestamp": (None, None), "text": " 나 "},
                                                         {"timestamp": (3.0, 4.0), "text": "  "}]}
    assert tr._onnx_rows(pipe, np.zeros(10, np.float32), "ko") == [(0.0, "가"), (0.0, "나")]
    assert tr._onnx_rows(lambda a, **k: {"text": " 전체 "}, None, "ko") == [(0.0, "전체")]


def test_stale_export_folders_are_cleaned(tmp_path):
    import os
    import time
    old, fresh, model = tmp_path / "tmpold", tmp_path / "tmpnew", tmp_path / "ready-model"
    for d in (old, fresh, model):
        (d / "model").mkdir(parents=True)
        (d / "model" / "w.bin").write_bytes(b"x")
    past = time.time() - 5 * 3600
    for f in (old / "model" / "w.bin", old / "model", old, model / "model" / "w.bin"):
        os.utime(f, (past, past))
    tr._clean_stale_exports(tmp_path)
    assert not old.exists() and fresh.exists() and model.exists()


def test_doctor_reports_and_skips_test_without_models(monkeypatch, tmp_path):
    for name in ("KO_DIR", "KO_MLX_DIR", "KO_OV_DIR", "KO_ONNX_DIR", "LARGE_V3_OV_DIR",
                 "LARGE_V3_TURBO_OV_DIR"):
        monkeypatch.setattr(tr, name, tmp_path / name)
    monkeypatch.setattr(tr, "_WHISPER_ONNX_DIRS", {k: tmp_path / k for k in tr._WHISPER_ONNX_DIRS})
    monkeypatch.setattr(tr, "gpu_backend", lambda gpu=None: "cpu")
    report = tr.doctor(run_test=True)
    assert "자동 선택: cpu" in report and "준비된 모델: 없음" in report and "건너뜀" in report


def test_install_rejects_conflicting_gpu_options():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "install.py"), "--amd-gpu", "--intel-gpu"],
                       capture_output=True, text=True)
    # 운영체제에 따라 먼저 걸리는 검사가 다름 (macOS에서는 --intel-gpu 자체가 거부됨)
    assert r.returncode == 2 and any(m in r.stderr for m in ("Windows only", "Choose one", "requires x86-64"))


# ---------- Apple Silicon (MLX) ----------
def test_hf_to_mlx_weight_names():
    assert tr._hf_to_mlx_name("model.encoder.layers.3.self_attn.q_proj.weight") == "encoder.blocks.3.attn.query.weight"
    assert tr._hf_to_mlx_name("model.decoder.layers.0.encoder_attn.k_proj.weight") == "decoder.blocks.0.cross_attn.key.weight"
    assert tr._hf_to_mlx_name("model.decoder.layers.1.encoder_attn_layer_norm.bias") == "decoder.blocks.1.cross_attn_ln.bias"
    assert tr._hf_to_mlx_name("model.encoder.layers.2.fc2.weight") == "encoder.blocks.2.mlp2.weight"
    assert tr._hf_to_mlx_name("model.encoder.layers.2.final_layer_norm.weight") == "encoder.blocks.2.mlp_ln.weight"
    assert tr._hf_to_mlx_name("model.decoder.embed_tokens.weight") == "decoder.token_embedding.weight"
    assert tr._hf_to_mlx_name("model.decoder.embed_positions.weight") == "decoder.positional_embedding"
    assert tr._hf_to_mlx_name("model.encoder.layer_norm.weight") == "encoder.ln_post.weight"
    assert tr._hf_to_mlx_name("model.decoder.layer_norm.bias") == "decoder.ln.bias"
    assert tr._hf_to_mlx_name("model.encoder.embed_positions.weight") is None
    assert tr._hf_to_mlx_name("proj_out.weight") is None


def test_clips_in_merges_close_speech_and_skips_silence():
    spans = [(1.0, 2.0), (3.0, 4.0), (20.0, 25.0), (40.0, 50.0)]
    # 조각 [0, 30] 기준 상대 시간. 1초 틈은 합치고 16초 무음은 건너뜀, 조각 밖 구간은 버림
    rounded = lambda clips: [[round(x, 3) for x in c] for c in clips]  # noqa: E731
    assert rounded(tr._clips_in(spans, 0.0, 30.0)) == [[0.8, 4.2], [19.8, 25.2]]
    assert rounded(tr._clips_in(spans, 30.0, 45.0)) == [[9.8, 15.0]]


def test_run_mlx_offsets_clips_and_recovers_loops(monkeypatch):
    import types
    import numpy as np
    loop = "같은 말을 계속 반복합니다 " * 8
    calls = []

    def fake_rows(model, audio, lang, **options):
        calls.append((len(audio) / tr.SR, options))
        return [(0.5, "재시도 성공")] if options else [(0.5, loop.strip())]

    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", types.SimpleNamespace(float16="f16"))
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.ModuleType("mlx_whisper"))
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe",
                        types.SimpleNamespace(ModelHolder=types.SimpleNamespace(get_model=lambda *a: None)))
    monkeypatch.setattr(tr, "_decode_from", lambda audio, start: (np.zeros(60 * tr.SR, np.float32), 70.0))
    monkeypatch.setattr(tr, "_speech_spans", lambda wav, sr: [(10.0, 12.0), (40.0, 41.0)])
    monkeypatch.setattr(tr, "_mlx_rows", fake_rows)
    progress = []
    rows = list(tr.run_mlx("a.m4a", "m", start_at=10.0, on_progress=progress.append))
    # 말소리 구간마다 따로 변환, 시각 = 이어하기 시작 + 구간 시작 + 줄 시작
    assert [round(t, 1) for t, _ in rows] == [20.3, 50.3]
    assert [text for _, text in rows] == ["재시도 성공", "재시도 성공"]
    assert progress[-1] == 1.0


def test_mlx_start_failure_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(tr, "gpu_backend", lambda gpu=None: "mlx")
    monkeypatch.setattr(tr, "ensure_ko_mlx_model", lambda: (_ for _ in ()).throw(tr._MLXStartError("no mlx")))
    monkeypatch.setattr(tr, "ensure_ko_model", lambda: "ct2")
    monkeypatch.setattr(tr, "run_whisper", lambda *a, **k: iter([(0.0, "CPU로 변환")]))
    assert list(tr.run_ko("a.m4a")) == [(0.0, "CPU로 변환")]


def test_whisper_engine_uses_mlx_repo_on_apple_silicon(monkeypatch):
    seen = []
    monkeypatch.setattr(tr, "gpu_backend", lambda gpu=None: "mlx")
    monkeypatch.setattr(tr, "run_mlx", lambda audio, model, lang, **k: seen.append(model) or iter([(0.0, "ok")]))
    monkeypatch.setattr(tr, "run_whisper", lambda *a, **k: seen.append("cpu") or iter([(0.0, "cpu")]))
    list(tr.transcribe("a.m4a", "whisper", "en", "small"))
    list(tr.transcribe("a.m4a", "whisper", "en", "/local/ct2-model"))   # MLX 변환본이 없는 로컬 모델은 CPU
    assert seen == ["mlx-community/whisper-small-mlx", "cpu"]
