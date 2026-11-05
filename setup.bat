@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo  녹음 변환기 설치 (Claude 데스크톱 연동)
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [오류] Python이 없어요. https://www.python.org 에서 설치할 때
  echo        "Add python.exe to PATH"를 꼭 체크해주세요.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] 가상환경 만드는 중...
  python -m venv .venv || goto :fail
)
set PY=.venv\Scripts\python.exe

echo [2/5] 패키지 설치 중... (몇 분 걸려요)
%PY% -m pip install --upgrade pip -q || goto :fail
%PY% -m pip install -q faster-whisper "mcp[cli]" huggingface_hub "transformers>=4.40,<5" pypdf markdown || goto :fail

echo [3/5] PyTorch(CPU) 설치 중... (한국어 모델 변환에만 쓰여요)
%PY% -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu || goto :fail

echo [4/5] 한국어 모델 내려받고 변환 중... (최초 1회, 5~10분)
%PY% -c "import transcribe; transcribe.ensure_ko_model()" || goto :fail

echo [5/5] Claude 데스크톱 앱에 등록 중...
%PY% register_claude.py || goto :fail

echo.
echo 설치 끝! Claude 데스크톱 앱을 완전히 종료했다가 다시 켠 뒤
echo "다운로드 폴더에 있는 최근 녹음 요약해줘" 라고 말해보세요.
pause
exit /b 0

:fail
echo.
echo [오류] 위 메시지를 Claude에게 보여주면 고쳐줄 거예요.
pause
exit /b 1
