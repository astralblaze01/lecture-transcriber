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
    if head[4:8] == b"ftyp":  # m4a, mp4, 3gp 등
        return AUDIO, ".m4a" if head[8:11] in (b"M4A", b"M4B") else ".mp4"
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
    return {"dir": out_dir, "audio": found[AUDIO], "pdf": found[PDF], "images": found[IMAGE]}


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
:root{--ink:#222;--muted:#666;--line:#e3ddd2;--paper:#fffdf8;--mark:#fff1a8;--key-bg:#fff4e5;--key:#c2410c}
@media (prefers-color-scheme:dark){:root{--ink:#e8e4dc;--muted:#a8a29a;--line:#3a3631;--paper:#1c1a17;--mark:#5c4d0f;--key-bg:#3a2a18;--key:#fb923c}}
body{background:var(--paper);color:var(--ink);font-family:"Noto Serif KR","Nanum Myeongjo",Batang,serif;
  line-height:1.85;font-size:17px;max-width:760px;margin:0 auto;padding:48px 20px 96px;word-break:keep-all}
h1{font-size:2em;line-height:1.3;margin:0 0 .3em}
h2{font-size:1.45em;margin:2.4em 0 .6em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.15em;margin:1.8em 0 .4em}
h1,h2,h3{font-family:"Pretendard","Malgun Gothic",sans-serif}
mark{background:var(--mark);color:inherit;padding:0 .15em;border-radius:3px}
strong{color:var(--key)}
blockquote{margin:1.4em 0;padding:.9em 1.1em;background:var(--key-bg);border-left:4px solid var(--key);border-radius:0}
blockquote p{margin:.3em 0}
code{font-family:Consolas,monospace;font-size:.9em;background:var(--line);padding:.1em .3em;border-radius:3px}
pre{background:var(--line);padding:1em;overflow-x:auto;border-radius:6px}
pre code{background:none;padding:0}
table{border-collapse:collapse;margin:1.2em 0;width:100%}
th,td{border:1px solid var(--line);padding:.45em .7em;text-align:left}
.src{color:var(--muted);font-size:.8em;font-family:"Pretendard","Malgun Gothic",sans-serif}
@media print{body{max-width:none;padding:0;font-size:11pt;background:#fff;color:#000}h2{break-before:page}
  mark{background:#fff1a8}blockquote{background:#fff4e5}}
"""


def markdown_to_book_html(md_text: str, title: str) -> str:
    import markdown

    # ==강조== → <mark>, 코드 블록 안은 건드리지 않음
    parts = re.split(r"(```.*?```|`[^`]*`)", md_text, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"==(.+?)==", r"<mark>\1</mark>", parts[i])
    body = markdown.markdown("".join(parts), extensions=["tables", "fenced_code", "sane_lists"])
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_html.escape(title)}</title><style>{BOOK_CSS}</style></head>"
            f"<body>{body}</body></html>")
