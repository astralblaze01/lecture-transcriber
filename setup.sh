#!/bin/bash
# 녹음 변환기 설치 (macOS, Claude 데스크톱 연동)
# install.py --register-claude 를 한 번에 실행하는 단축 스크립트.
# Apple Silicon(M1~)이면 MLX로 GPU를 쓰고, 인텔 Mac은 CPU(faster-whisper)로 동작해요.
cd "$(dirname "$0")"

PYTHON=""
for c in python3.13 python3.12 python3.11 python3.10 python3.14 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PYTHON="$c"; break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "[오류] Python 3.10 이상이 없어요. https://www.python.org 에서 설치하거나"
  echo "       Homebrew로 'brew install python' 해주세요."
  exit 1
fi

if ! "$PYTHON" install.py --register-claude "$@"; then
  echo
  echo "[오류] 위 메시지를 Claude에게 보여주면 고쳐줄 거예요."
  exit 1
fi
