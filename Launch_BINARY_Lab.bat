@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" verify_binary_release.py
) else if exist "%USERPROFILE%\.bci_studio\venv\Scripts\python.exe" (
  "%USERPROFILE%\.bci_studio\venv\Scripts\python.exe" verify_binary_release.py
) else (
  where py >nul 2>&1
  if errorlevel 1 (python verify_binary_release.py) else (py -3 verify_binary_release.py)
)
if errorlevel 1 (
  echo Binary release integrity check failed. Review the listed files before lab validation.
  pause
  exit /b 1
)
call Launch_BCI_3D_Studio.bat %*
exit /b %ERRORLEVEL%
