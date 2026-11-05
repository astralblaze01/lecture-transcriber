"""모델 없이 돌아가는 핵심 동작 테스트: pytest -q"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lecture_tools as lt  # noqa: E402
import transcribe as tr  # noqa: E402


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
    (tmp_path / "rec.ko.txt").write_text("옛 결과\n", encoding="utf-8")
    (tmp_path / "rec.ko.txt.part").write_text("[00:05:00] 옛 조각\n", encoding="utf-8")
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
