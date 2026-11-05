"""
강의 노트 처리: 삼성노트(.sdocx) 풀기, PDF 텍스트 추출, 책 문서(md/html) 만들기.
transcribe_mcp.py가 불러서 씀.
"""
import html as _html
import re
import zipfile
from pathlib import Path

AUDIO, PDF, IMAGE = "audio", "pdf", "image"


def sniff(head: bytes):
    """확장자 대신 파일 앞부분(시그니처)으로 종류 판별. 삼성노트는 내부 파일 이름이 제각각이라서."""
    if head.startswith(b"%PDF"):
        return PDF, ".pdf"
    if head[4:8] == b"ftyp":  # m4a, mp4, 3gp 등 (삼성노트 녹음은 3gp4 브랜드의 m4a)
        return AUDIO, ".mp4" if head[8:12] in (b"mp41", b"mp42", b"isom", b"qt  ") else ".m4a"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return AUDIO, ".mp3"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return AUDIO, ".wav"
    if head.startswith(b"#!AMR"):
        return AUDIO, ".amr"
    if head.startswith(b"OggS"):
        return AUDIO, ".ogg"
    if head.startswith(b"\x89PNG"):
        return IMAGE, ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return IMAGE, ".jpg"
    return None, None


def extract_note(note: Path) -> dict:
    """sdocx(압축 파일) 안의 녹음·PDF를 '<이름>_추출' 폴더로 꺼냄. 이미 꺼냈으면 재사용."""
    out_dir = note.with_name(note.stem + "_추출")
    if not zipfile.is_zipfile(note):
        with note.open("rb") as f:
            head = f.read(16)
        raise ValueError(f"압축 형식이 아니에요 (앞부분: {head!r}). 삼성노트 앱에서 PDF/녹음을 직접 내보내 주세요.")

    out_dir.mkdir(exist_ok=True)
    found = {AUDIO: [], PDF: [], IMAGE: 0}
    with zipfile.ZipFile(note) as z:
        for i, info in enumerate(sorted(z.infolist(), key=lambda x: x.filename)):
            if info.is_dir() or info.file_size < 64:
                continue
            with z.open(info) as f:
                head = f.read(16)
            kind, ext = sniff(head)
            if kind == IMAGE:
                found[IMAGE] += 1
                continue
            if kind not in (AUDIO, PDF):
                continue
            # 압축 내부 경로는 버리고 안전한 이름만 사용 (경로 조작 방지)
            base = re.sub(r"[^\w.\-가-힣]", "_", Path(info.filename).name) or "file"
            stem = Path(base).stem
            target = out_dir / f"{i:02d}_{stem}{ext}"
            if not target.exists() or target.stat().st_size != info.file_size:
                with z.open(info) as src, target.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            found[kind].append(target)
    # 노트 안 파일 이름 순서는 녹음 순서와 달라서, 녹음 시각 순으로 정렬
    found[AUDIO].sort(key=_recorded_at)
    return {"dir": out_dir, "audio": found[AUDIO], "pdf": found[PDF], "images": found[IMAGE]}


def _recorded_at(path: Path) -> str:
    try:
        import av  # faster-whisper와 함께 설치됨
        with av.open(str(path)) as c:
            return c.metadata.get("creation_time", "") or path.name
    except Exception:  # noqa: BLE001
        return path.name


def _parse_pages(spec: str, n: int):
    if not spec:
        return list(range(1, n + 1))
    pages = []
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a or 1), int(b or n) + 1))
        elif part:
            pages.append(int(part))
    return [p for p in pages if 1 <= p <= n]


def pdf_text(pdf: Path, pages: str = "") -> tuple[int, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    n = len(reader.pages)
    blocks = []
    for p in _parse_pages(pages, n):
        try:
            t = (reader.pages[p - 1].extract_text() or "").strip()
        except Exception as e:  # noqa: BLE001
            t = f"(추출 실패: {e})"
        if len(t) < 15:
            t += "\n(텍스트가 거의 없음 — 그림/도표 위주 슬라이드일 수 있음)"
        blocks.append(f"=== p.{p} ===\n{t}")
    return n, "\n\n".join(blocks)


BOOK_CSS = """
:root{--ink:#222;--muted:#666;--line:#e3ddd2;--paper:#fffdf8;--mark:#fff1a8;--key-bg:#fff4e5;--key:#c2410c;--note-bg:#f1f4f8;--note:#64748b}
@media (prefers-color-scheme:dark){:root{--ink:#e8e4dc;--muted:#a8a29a;--line:#3a3631;--paper:#1c1a17;--mark:#5c4d0f;--key-bg:#3a2a18;--key:#fb923c;--note-bg:#252a31;--note:#94a3b8}}
body{background:var(--paper);color:var(--ink);font-family:"Noto Serif KR","Nanum Myeongjo",Batang,serif;
  line-height:1.85;font-size:17px;max-width:760px;margin:0 auto;padding:48px 20px 96px;word-break:keep-all}
h1{font-size:2em;line-height:1.3;margin:0 0 .3em}
h2{font-size:1.45em;margin:2.4em 0 .6em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.15em;margin:1.8em 0 .4em}
h1,h2,h3{font-family:"Pretendard","Malgun Gothic",sans-serif}
mark{background:var(--mark);color:inherit;padding:0 .15em;border-radius:3px}
strong{color:var(--key)}
blockquote{margin:1.4em 0;padding:.9em 1.1em;background:var(--note-bg);border-left:4px solid var(--note);border-radius:0}
blockquote.exam{background:var(--key-bg);border-left-color:var(--key)}
blockquote p{margin:.3em 0}
.math.display{overflow-x:auto;overflow-y:hidden;margin:1em 0;text-align:center}
td .math.display,th .math.display{margin:0;text-align:left}
code{font-family:Consolas,monospace;font-size:.9em;background:var(--line);padding:.1em .3em;border-radius:3px}
pre{background:var(--line);padding:1em;overflow-x:auto;border-radius:6px}
pre code{background:none;padding:0}
table{border-collapse:collapse;margin:1.2em 0;width:100%}
th,td{border:1px solid var(--line);padding:.45em .7em;text-align:left}
.src{color:var(--muted);font-size:.8em;font-family:"Pretendard","Malgun Gothic",sans-serif}
@media print{body{max-width:none;padding:0;font-size:11pt;background:#fff;color:#000}
  mark{background:#fff1a8}blockquote{background:#f1f4f8}blockquote.exam{background:#fff4e5}
  h2,h3{break-after:avoid}table,blockquote,pre,.math.display{break-inside:avoid}}
"""


KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
_MATH_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
_MATH_INLINE = re.compile(r"(?<![\\$])\$(?![\s$])([^$\n]+?)(?<!\s)\$(?!\d)")


def markdown_to_book_html(md_text: str, title: str) -> str:
    """책 원고(마크다운)를 인쇄용 HTML로. '==텍스트=='는 형광펜, $...$/$$...$$는 KaTeX 수식,
    '> **시험 포인트** ...' 인용은 주황 박스, 그 밖의 인용은 회색 메모 박스로 표시."""
    import markdown

    # 코드 → 수식 순서로 보호한 뒤 마크다운 변환 (수식 안의 _ * \\ 가 망가지지 않게)
    saved = []

    def keep(kind):
        def _f(m):
            saved.append((kind, m.group(1) if kind != "code" else m.group(0)))
            return f"KEEPTOKEN{len(saved) - 1}END"
        return _f

    text = re.sub(r"```.*?```|`[^`]*`", keep("code"), md_text, flags=re.S)
    text = _MATH_DISPLAY.sub(keep("display"), text)
    text = _MATH_INLINE.sub(keep("inline"), text)
    text = re.sub(r"==(.+?)==", r"<mark>\1</mark>", text)
    # Python-Markdown은 문단 바로 다음 줄의 목록을 목록으로 안 봐서, 그 앞에 빈 줄을 넣어 줌
    lines, fixed = text.split("\n"), []
    for ln in lines:
        is_item = re.match(r"\s*(?:[-*+]|\d+\.)\s", ln)
        prev = fixed[-1] if fixed else ""
        if is_item and prev.strip() and not re.match(r"\s*(?:[-*+]|\d+\.)\s", prev) \
                and not prev.lstrip().startswith(("|", ">")) and not ln.startswith((" ", "\t")):
            fixed.append("")
        fixed.append(ln)
    text = "\n".join(fixed)

    def restore_code(t):
        return re.sub(r"KEEPTOKEN(\d+)END",
                      lambda m: saved[int(m.group(1))][1] if saved[int(m.group(1))][0] == "code" else m.group(0), t)

    body = markdown.markdown(restore_code(text), extensions=["tables", "fenced_code", "sane_lists"])

    def restore_math(m):
        kind, tex = saved[int(m.group(1))]
        tag = "div" if kind == "display" else "span"
        return f'<{tag} class="math {kind}">{_html.escape(tex.strip())}</{tag}>'

    body = re.sub(r"KEEPTOKEN(\d+)END", restore_math, body)
    body = re.sub(r"<p>(<div class=\"math display\">.*?</div>)</p>", r"\1", body, flags=re.S)
    body = body.replace("<blockquote>\n<p><strong>시험 포인트", '<blockquote class="exam">\n<p><strong>시험 포인트')

    katex = (f'<link rel="stylesheet" href="{KATEX_CDN}/katex.min.css">'
             f'<script defer src="{KATEX_CDN}/katex.min.js"></script>'
             "<script>addEventListener('DOMContentLoaded',()=>{if(!window.katex)return;"
             "document.querySelectorAll('.math').forEach(e=>{try{katex.render(e.textContent,e,"
             "{displayMode:e.classList.contains('display'),throwOnError:false})}catch(_){}})})</script>"
             if saved and any(k != "code" for k, _ in saved) else "")
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_html.escape(title)}</title><style>{BOOK_CSS}</style>{katex}</head>"
            f"<body>{body}</body></html>")
