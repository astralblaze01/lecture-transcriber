@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run setup.bat or "python install.py" first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe.py" --choose-model %*
set "result=%errorlevel%"
pause
exit /b %result%
