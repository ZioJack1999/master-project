@echo off
setlocal
cd /d "%~dp0"
if "%X24_DECODER_PYTHON%"=="" (
  if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "X24_DECODER_PYTHON=%USERPROFILE%\miniconda3\python.exe"
  ) else if exist "C:\tools\Anaconda3\envs\multi_nn\python.exe" (
    set "X24_DECODER_PYTHON=C:\tools\Anaconda3\envs\multi_nn\python.exe"
  ) else if exist "%USERPROFILE%\.bci_studio\venv\Scripts\python.exe" (
    set "X24_DECODER_PYTHON=%USERPROFILE%\.bci_studio\venv\Scripts\python.exe"
  ) else (
    set "X24_DECODER_PYTHON=%USERPROFILE%\miniconda3\python.exe"
  )
)
if not exist "%X24_DECODER_PYTHON%" (
  echo Existing 64-bit conda Python was not found at "%X24_DECODER_PYTHON%".
  exit /b 1
)
set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Library\bin;%PATH%"
set "PYTHONPATH=%~dp0"
"%X24_DECODER_PYTHON%" -u run_x24.py --task binary %*
exit /b %ERRORLEVEL%
