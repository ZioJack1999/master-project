@echo off
setlocal
cd /d "%~dp0"
set "X24_PYTHON=%USERPROFILE%\miniconda3\envs\mastersp\python.exe"
if not exist "%X24_PYTHON%" (
  echo Existing mastersp Python was not found at "%X24_PYTHON%".
  exit /b 1
)
set "PATH=%USERPROFILE%\miniconda3\envs\mastersp;%USERPROFILE%\miniconda3\envs\mastersp\Library\bin;%PATH%"
set "PYTHONPATH=%~dp0.runtime\x24-lsl32;%~dp0"
set "PYTHONNOUSERSITE=1"
"%X24_PYTHON%" -u x24_lsl_bridge.py %*
exit /b %ERRORLEVEL%
