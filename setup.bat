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
%PY% -m pip install -q faster-whisper "mcp[cli]" huggingface_hub "transformers>=5.14" pypdf markdown "av<19" || goto :fail

echo [3/5] PyTorch(CPU) 설치 중... (한국어 모델 변환에만 쓰여요)
%PY% -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu || goto :fail

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo [GPU] NVIDIA GPU found - installing CUDA libraries...
  %PY% -m pip install -q nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10" || echo [GPU] install failed - continuing with CPU.
)

powershell -NoProfile -Command "if ((Get-CimInstance Win32_VideoController).Name -match 'Intel') { exit 0 } else { exit 1 }" >nul 2>nul
if not errorlevel 1 (
  echo [GPU] Intel GPU found - installing OpenVINO...
  %PY% -m pip install -q torchvision --index-url https://download.pytorch.org/whl/cpu
  %PY% -m pip install -q openvino-genai "optimum-intel[openvino]" "transformers>=4.51,<5" && %PY% -c "import transcribe as t; d=t.ov_device(); print('[GPU] OpenVINO device:', d); d and t.ensure_ko_ov_model()" || echo [GPU] OpenVINO setup failed - continuing with CPU.
)

powershell -NoProfile -Command "$n=(Get-CimInstance Win32_VideoController).Name -join ' '; if (($n -match 'AMD|Radeon') -and -not ($n -match 'NVIDIA|Intel')) { exit 0 } else { exit 1 }" >nul 2>nul
if not errorlevel 1 (
  echo [GPU] AMD GPU found - installing DirectML...
  %PY% -m pip uninstall -y -q onnxruntime
  %PY% -m pip install -q onnxruntime-directml optimum-onnx "transformers>=4.51,<4.58" && %PY% -c "import transcribe as t; ok=t.directml_available(); print('[GPU] DirectML:', ok); ok and t.ensure_onnx_model(t.KO_REPO, t.KO_ONNX_DIR)" || echo [GPU] DirectML setup failed - continuing with CPU.
)

echo [4/5] 한국어 모델 내려받고 변환 중... (최초 1회, 5~10분)
%PY% -c "import transcribe; transcribe.ensure_ko_model()" || goto :fail

echo [5/5] Claude 데스크톱 앱에 등록 중...
%PY% register_claude.py || goto :fail

pause
exit /b 0

:fail
echo.
echo [오류] 위 메시지를 Claude에게 보여주면 고쳐줄 거예요.
pause
exit /b 1
