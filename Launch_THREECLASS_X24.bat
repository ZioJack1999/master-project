@echo off
setlocal
cd /d "%~dp0"
set "X24_DECODER_PYTHON=%USERPROFILE%\miniconda3\python.exe"
if not exist "%X24_DECODER_PYTHON%" (
  echo Existing 64-bit conda Python was not found at "%X24_DECODER_PYTHON%".
  exit /b 1
)
set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Library\bin;%PATH%"
set "PYTHONPATH=%~dp0"
"%X24_DECODER_PYTHON%" -u run_x24.py --task threeclass %*
exit /b %ERRORLEVEL%
