"""
Claude 데스크톱 앱 설정(claude_desktop_config.json)에 transcriber MCP 서버를 등록.
기존 설정은 .bak으로 백업하고, 다른 MCP 서버 설정은 그대로 둠.
실행에 사용한 가상환경 Python 경로를 등록한다.
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE / "transcribe_mcp.py"
NAME = "transcriber"


def config_dirs():
    if sys.platform == "darwin":
        return [Path.home() / "Library" / "Application Support" / "Claude"]
    if sys.platform != "win32":
        return []
    dirs = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "Claude")
    # Microsoft Store(MSIX) 설치판은 설정 위치가 다름
    local = os.environ.get("LOCALAPPDATA")
    if local:
        for p in glob.glob(os.path.join(local, "Packages", "Claude_*", "LocalCache", "Roaming", "Claude")):
            dirs.append(Path(p))
    existing = [d for d in dirs if d.is_dir()]
    return existing or dirs[:1]


def main():
    entry = {
        "command": sys.executable,
        "args": [str(SERVER)],
        "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    }
    targets = config_dirs()
    if not targets:
        sys.exit("Claude 데스크톱 설정 위치를 찾을 수 없습니다. 사용 중인 MCP 클라이언트 설정에 transcribe_mcp.py를 등록하세요.")

    for d in targets:
        d.mkdir(parents=True, exist_ok=True)
        cfg_path = d / "claude_desktop_config.json"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8") or "{}")
            except json.JSONDecodeError:
                sys.exit(f"설정 파일이 올바른 JSON이 아니에요: {cfg_path}\n직접 확인 후 다시 실행해주세요.")
            shutil.copy2(cfg_path, cfg_path.with_suffix(".json.bak"))
        cfg.setdefault("mcpServers", {})[NAME] = entry
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"등록 완료: {cfg_path}")

    quit_hint = "메뉴 막대 Claude → Quit Claude(⌘Q)" if sys.platform == "darwin" else "트레이 아이콘 → 종료"
    print(f"\n설치 끝! Claude 데스크톱 앱을 완전히 종료({quit_hint})한 뒤 다시 켜고,")
    print("새 채팅에서 '다운로드 폴더에 있는 최근 녹음 요약해줘'라고 말해보세요.")


if __name__ == "__main__":
    main()
