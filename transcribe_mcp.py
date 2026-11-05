"""
녹음 변환 MCP 서버 — Claude 데스크톱 앱이 이 PC에서 직접 녹음을 텍스트로 바꿀 수 있게 해줌.

도구
  find_recordings      : 다운로드·바탕화면·문서 등에서 최근 녹음/영상 파일 찾기
  start_transcription  : 변환 작업 시작 (긴 녹음도 타임아웃 안 나게 백그라운드로 실행)
  get_transcription    : 진행률 확인 / 완료되면 변환된 텍스트 받기
  prepare_lecture_note : 삼성노트(.sdocx)·PDF를 풀어서 녹음 변환까지 한 번에 시작
  read_pdf_text        : 강의 PDF의 페이지별 텍스트
  save_book            : Claude가 쓴 책 원고를 .md + 보기 좋은 .html로 저장

등록은 setup.bat이 해줌. 수동 등록은 README 참고.
"""
import asyncio
import os
import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcribe as tr  # noqa: E402
import lecture_tools as lt  # noqa: E402

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".oga", ".opus", ".flac",
              ".wma", ".amr", ".3gp", ".webm", ".mp4", ".mov", ".mkv", ".avi"}
NOTE_EXTS = {".sdocx", ".pdf"}
MAX_WAIT = 50  # 도구 호출 하나가 너무 오래 걸리지 않게 (클라이언트 타임아웃 회피)

mcp = FastMCP(
    "transcriber",
    instructions=(
        "사용자 PC의 녹음·영상과 강의 노트(삼성노트 .sdocx, PDF)를 로컬에서 처리하는 도구입니다.\n"
        "[녹음 변환] 경로를 모르면 find_recordings로 찾고, 애매하면 사용자에게 확인하세요. "
        "start_transcription 후 get_transcription을 status가 done이 될 때까지 반복 호출하세요. "
        "CPU라 1시간 녹음이면 수십 분 걸릴 수 있으니 진행률을 짧게 알려주세요. "
        "한국어는 engine='ko'(기본), 다른 언어는 engine='whisper'와 language 코드를 쓰세요.\n"
        "[강의를 책으로] 사용자가 노트/PDF와 녹음을 바탕으로 정리·책·요약을 원하면: "
        "1) prepare_lecture_note로 노트를 풀고 녹음 변환을 시작, "
        "2) 기다리는 동안 read_pdf_text로 슬라이드 내용 파악, "
        "3) get_transcription(timestamps=true)으로 강의 녹취 전체를 받음(offset으로 끝까지), "
        "4) 슬라이드 순서를 뼈대로 교수님 설명을 녹여 교재처럼 풀어 쓴 한국어 마크다운 원고 작성, "
        "5) save_book으로 저장하고 저장 경로를 알려주세요.\n"
        "원고 규칙: '# 제목' 하나, 슬라이드 묶음마다 '## 장', 세부는 '###'. 녹취의 말투·반복·잡담은 걷어내고 문어체로. "
        "교수님이 강조한 말(중요하다, 시험, 꼭 기억, 반복 설명 등)이 붙은 내용은 '==형광펜==', 핵심 용어는 '**굵게**', "
        "시험 포인트는 '> **시험 포인트** ...' 인용 블록으로 표시. 슬라이드에 없지만 말로만 한 설명도 빠뜨리지 말고, "
        "녹취에서 인식이 틀린 게 분명한 용어는 슬라이드 용어로 바로잡으세요. 장 끝에 근거 슬라이드 범위를 '<p class=\"src\">p.3–7</p>'로 남기세요."
    ),
)


# ---------- 작업 관리 ----------
jobs: dict[str, dict] = {}
job_queue: "queue.Queue[str]" = queue.Queue()


def _worker():
    # 한 번에 하나씩만 변환 (노트북 CPU를 여러 작업이 나눠 쓰면 다 같이 느려짐)
    while True:
        jid = job_queue.get()
        job = jobs[jid]
        job["status"] = "running"
        job["started"] = time.time()
        try:
            def on_progress(p):
                job["progress"] = p

            # 중간에 끊긴 .part가 있으면 그 지점부터 이어서 함. 실패해도 .part는 남겨 다음에 이어감.
            out = tr.transcribe_to_file(job["file"], job["engine"], job["language"],
                                        job["model"], job["gpu"], on_progress, force=job["force"])
            job.update(status="done", progress=1.0, output=str(out), finished=time.time())
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=f"{type(e).__name__}: {e} (다시 시작하면 끊긴 지점부터 이어서 해요)",
                       finished=time.time())
        finally:
            job_queue.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _clean_path(p: str) -> Path:
    p = p.strip().strip('"').strip("'")
    return Path(os.path.expandvars(os.path.expanduser(p)))


def _duration(path: Path):
    try:
        import av  # faster-whisper와 함께 설치됨
        with av.open(str(path)) as c:
            return c.duration / 1_000_000 if c.duration else None
    except Exception:  # noqa: BLE001
        return None


def _default_folders():
    home = Path.home()
    names = ["Downloads", "Desktop", "Documents", "Music", "Videos",
             "Documents/Sound recordings", "Documents/녹음"]
    roots = [home]
    onedrive = os.environ.get("OneDrive")
    if onedrive:
        roots.append(Path(onedrive))
    seen, out = set(), []
    for r in roots:
        for n in names + ["바탕 화면", "문서"]:
            d = r / n
            if d.is_dir() and d.resolve() not in seen:
                seen.add(d.resolve())
                out.append(d)
    return out


def _walk(folder: Path, depth: int):
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and Path(entry.name).suffix.lower() in (AUDIO_EXTS | NOTE_EXTS):
                yield Path(entry.path)
            elif entry.is_dir() and depth > 0 and not entry.name.startswith("."):
                yield from _walk(Path(entry.path), depth - 1)
    except (PermissionError, OSError):
        return


def _status_text(job, timestamps, offset, max_chars):
    elapsed = (job.get("finished") or time.time()) - job.get("started", time.time())
    head = [f"job_id: {job['id']}", f"file: {job['file']}", f"engine: {job['engine']}",
            f"status: {job['status']}"]

    if job["status"] in ("queued", "running"):
        p = job.get("progress", 0.0)
        head.append(f"progress: {p * 100:.0f}%")
        if job["status"] == "running":
            head.append(f"elapsed: {tr.fmt(elapsed)}")
            if p > 0.03:
                head.append(f"eta: 약 {tr.fmt(elapsed / p - elapsed)}")
            elif job["engine"] == "ko" and not (tr.KO_DIR / "model.bin").exists():
                head.append("note: 한국어 모델 최초 변환 중일 수 있음 (몇 분)")
        else:
            head.append(f"queue_position: {list(job_queue.queue).index(job['id']) + 1 if job['id'] in job_queue.queue else '?'}")
        head.append("→ 잠시 후 get_transcription을 다시 호출하세요.")
        return "\n".join(head)

    if job["status"] == "error":
        head.append(f"error: {job['error']}")
        return "\n".join(head)

    text = Path(job["output"]).read_text(encoding="utf-8")
    if not timestamps:
        text = "\n".join(l.split("] ", 1)[1] if l.startswith("[") and "] " in l else l
                         for l in text.splitlines())
    total = len(text)
    chunk = text[offset:offset + max_chars]
    head.append(f"saved_to: {job['output']}")
    if job.get("cached"):
        head.append("cached: 이전에 변환해둔 결과를 재사용함 (다시 하려면 force=true)")
    else:
        head.append(f"took: {tr.fmt(elapsed)}")
    head.append(f"chars: {offset}–{offset + len(chunk)} / {total}")
    if offset + len(chunk) < total:
        head.append(f"→ 나머지는 offset={offset + len(chunk)}로 다시 호출")
    return "\n".join(head) + "\n\n----- transcript -----\n" + chunk


# ---------- 도구 ----------
@mcp.tool()
def find_recordings(folder: str = "", limit: int = 15) -> str:
    """최근 녹음/영상 파일과 강의 노트(.sdocx, .pdf)를 수정 시간 순으로 찾습니다.
    folder를 비우면 다운로드·바탕화면·문서·음악·동영상(OneDrive 포함)을 2단계 깊이까지 찾습니다."""
    folders = [_clean_path(folder)] if folder else _default_folders()
    files = {}
    for d in folders:
        if not d.is_dir():
            return f"폴더를 찾을 수 없어요: {d}"
        for f in _walk(d, 2):
            files[str(f)] = f
    if not files:
        return "녹음/영상 파일을 찾지 못했어요. 찾은 폴더: " + ", ".join(map(str, folders))

    ranked = sorted(files.values(), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]
    lines = []
    for f in ranked:
        st = f.stat()
        dur = _duration(f) if f.suffix.lower() in AUDIO_EXTS else None
        done = tr.finished_outputs(f)
        lines.append(
            f"- {f}\n  수정: {datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}, "
            f"크기: {st.st_size / 1e6:.1f}MB" + (f", 길이: {tr.fmt(dur)}" if dur else "")
            + (f", 변환됨: {', '.join(done)}" if done else "")
        )
    return "\n".join(lines)


@mcp.tool()
def start_transcription(file_path: str, engine: str = "ko", language: str = "ko",
                        model: str = "large-v3-turbo", gpu: bool | None = None,
                        force: bool = False) -> str:
    """녹음 파일을 텍스트로 변환하는 작업을 시작하고 job_id를 돌려줍니다.
    engine: 'ko'(한국어 특화, 기본) | 'whisper'(다국어, model로 크기 선택) | 'cohere'(GPU 권장)
    gpu: 비우면 지원 GPU 자동 사용, false면 CPU 강제. Intel GPU는 ko와 whisper large-v3/large-v3-turbo 지원.
    language: 'ko', 'en' 등. 이미 변환한 파일은 force=false면 저장된 결과를 바로 씁니다."""
    path = _clean_path(file_path)
    if not path.is_file():
        return f"파일이 없어요: {path}  (find_recordings로 경로를 확인하세요)"
    if engine not in tr.ENGINES:
        return f"engine은 {', '.join(tr.ENGINES)} 중 하나여야 해요."

    jid = uuid.uuid4().hex[:8]
    job = dict(id=jid, file=str(path), engine=engine, language=language,
               model=model, gpu=gpu, force=force, status="queued", progress=0.0)
    jobs[jid] = job

    out = tr.output_path(path, engine, model)
    if out.exists() and not force:
        job.update(status="done", progress=1.0, output=str(out), cached=True,
                   started=time.time(), finished=time.time())
        return f"job_id: {jid}\nstatus: done (이전 결과 재사용)\n→ get_transcription(job_id='{jid}')로 텍스트를 받으세요."

    # 같은 파일·엔진이 이미 돌고 있으면 그 작업을 알려줌
    for other in jobs.values():
        if (other is not job and other["file"] == job["file"] and other["engine"] == engine
                and tr.output_path(path, engine, other["model"]) == out
                and other["status"] in ("queued", "running")):
            del jobs[jid]
            return f"이미 진행 중인 작업이 있어요. job_id: {other['id']}"

    job_queue.put(jid)
    dur = _duration(path)
    return (f"job_id: {jid}\nstatus: queued\n녹음 길이: {tr.fmt(dur) if dur else '?'}\n"
            f"→ get_transcription(job_id='{jid}')로 진행 상황을 확인하세요.")


@mcp.tool()
async def get_transcription(job_id: str, wait_seconds: int = 40, timestamps: bool = True,
                            offset: int = 0, max_chars: int = 40000) -> str:
    """변환 작업의 진행률을 확인하거나, 완료됐으면 텍스트를 돌려줍니다.
    wait_seconds 동안(최대 50초) 완료를 기다렸다가 응답합니다.
    텍스트가 길면 max_chars 단위로 잘라서 주며, offset으로 이어서 받을 수 있습니다.
    요약만 할 거면 timestamps=false로 토큰을 아낄 수 있습니다."""
    job = jobs.get(job_id)
    if not job:
        return f"job_id {job_id}를 찾을 수 없어요. (서버가 재시작됐다면 start_transcription을 다시 호출하세요. 완료된 결과는 캐시돼 있어요.)"
    deadline = time.time() + max(0, min(wait_seconds, MAX_WAIT))
    while job["status"] in ("queued", "running") and time.time() < deadline:
        await asyncio.sleep(1)  # 이벤트 루프를 막지 않게
    return _status_text(job, timestamps, max(0, offset), max(1000, max_chars))


allowed_out: set[Path] = set()  # save_book이 쓸 수 있는 폴더 (처리한 노트가 있는 곳만)


def _remember_dir(d: Path):
    allowed_out.add(d.resolve())


@mcp.tool()
def prepare_lecture_note(note_path: str, engine: str = "ko", language: str = "ko",
                         model: str = "large-v3-turbo", gpu: bool | None = None,
                         force: bool = False) -> str:
    """강의 노트를 책으로 만들기 위한 준비를 한 번에 합니다.
    .sdocx(삼성노트)면 압축을 풀어 녹음과 PDF를 꺼내고, 녹음 변환 작업을 바로 시작합니다.
    .pdf면 같은 폴더에서 이름이 비슷한 녹음을 찾아 변환을 시작합니다.
    돌려준 job_id들로 get_transcription, PDF 경로로 read_pdf_text를 쓰세요."""
    path = _clean_path(note_path)
    if not path.is_file():
        return f"파일이 없어요: {path}  (find_recordings로 경로를 확인하세요)"

    ext = path.suffix.lower()
    if ext == ".sdocx":
        try:
            info = lt.extract_note(path)
        except Exception as e:  # noqa: BLE001
            return f"노트를 풀지 못했어요: {type(e).__name__}: {e}"
        out_dir, audios, pdfs = info["dir"], info["audio"], info["pdf"]
        lines = [f"추출 폴더: {out_dir}", f"그림 파일 {info['images']}개는 건너뜀"]
    elif ext == ".pdf":
        out_dir, pdfs = path.parent, [path]
        key = re.sub(r"\W+", "", path.stem.lower())[:12]
        audios = [f for f in path.parent.iterdir()
                  if f.suffix.lower() in AUDIO_EXTS and key and key in re.sub(r"\W+", "", f.stem.lower())]
        lines = [f"폴더: {out_dir}"]
    else:
        return "prepare_lecture_note는 .sdocx나 .pdf를 받아요. 녹음만 있으면 start_transcription을 쓰세요."

    _remember_dir(out_dir)
    _remember_dir(path.parent)

    for p in pdfs:
        try:
            n, _ = lt.pdf_text(p, "1")
            lines.append(f"PDF: {p}  ({n}쪽)")
        except Exception as e:  # noqa: BLE001
            lines.append(f"PDF: {p}  (읽기 실패: {e})")
    if not pdfs:
        lines.append("PDF: 없음 — 녹취만으로 정리하거나 사용자에게 PDF 위치를 물어보세요.")

    if not audios:
        lines.append("녹음: 없음 — 노트에 녹음이 안 들어있어요. 사용자에게 녹음 파일 위치를 물어보세요.")
    for a in audios:
        res = start_transcription(str(a), engine=engine, language=language, model=model, gpu=gpu, force=force)
        jid = re.search(r"job_id: (\w+)", res)
        dur = _duration(a)
        lines.append(f"녹음: {a}  (길이 {tr.fmt(dur) if dur else '?'})  → "
                     + (f"job_id {jid.group(1)}" if jid else res.replace("\n", " ")))
    if len(audios) > 1:
        lines.append("녹음이 여러 개예요. 파일 이름 순서가 강의 순서일 가능성이 높아요.")
    lines.append("다음: 변환을 기다리는 동안 read_pdf_text로 슬라이드를 읽으세요.")
    return "\n".join(lines)


@mcp.tool()
def read_pdf_text(pdf_path: str, pages: str = "", max_chars: int = 40000) -> str:
    """PDF의 페이지별 텍스트를 '=== p.N ===' 형식으로 돌려줍니다.
    pages 예: '' (전체), '1-10', '3,5,7-9'. 길면 잘리니 pages로 나눠 받으세요."""
    path = _clean_path(pdf_path)
    if not path.is_file():
        return f"파일이 없어요: {path}"
    try:
        n, text = lt.pdf_text(path, pages)
    except ImportError:
        return "pypdf가 설치돼 있지 않아요. setup.bat을 다시 실행해주세요."
    except Exception as e:  # noqa: BLE001
        return f"PDF를 읽지 못했어요: {type(e).__name__}: {e}"
    _remember_dir(path.parent)
    cut = len(text) > max_chars
    return (f"총 {n}쪽\n" + text[:max_chars]
            + ("\n\n(잘림 — pages를 좁혀서 다시 호출하세요)" if cut else ""))


@mcp.tool()
def save_book(folder: str, title: str, markdown: str) -> str:
    """Claude가 쓴 책 원고(마크다운)를 저장합니다. '==텍스트=='는 형광펜으로 표시됩니다.
    folder는 prepare_lecture_note/read_pdf_text로 다룬 노트의 폴더여야 합니다.
    <제목>.md 와 브라우저로 열어 보거나 PDF로 인쇄하기 좋은 <제목>.html 을 만듭니다."""
    d = _clean_path(folder).resolve()
    if d not in allowed_out:
        return ("이 폴더에는 저장할 수 없어요. 처리한 노트가 있는 폴더 중 하나를 쓰세요: "
                + ", ".join(map(str, sorted(allowed_out))) if allowed_out else
                "먼저 prepare_lecture_note나 read_pdf_text로 노트를 처리하세요.")
    safe = re.sub(r'[\\/:*?"<>|]', "_", title).strip()[:80] or "강의노트"
    md_path, html_path = d / f"{safe}.md", d / f"{safe}.html"
    md_path.write_text(markdown, encoding="utf-8")
    try:
        html_path.write_text(lt.markdown_to_book_html(markdown, title), encoding="utf-8")
    except ImportError:
        return f"원고는 저장했어요: {md_path}\nHTML 변환용 markdown 패키지가 없어요. setup.bat을 다시 실행해주세요."
    return (f"저장 완료\n- 책(브라우저로 열기, Ctrl+P로 PDF 저장 가능): {html_path}\n- 원고: {md_path}\n"
            f"- 형광펜 {markdown.count('==') // 2}곳, 시험 포인트 {markdown.count('시험 포인트')}곳")


if __name__ == "__main__":
    mcp.run()
