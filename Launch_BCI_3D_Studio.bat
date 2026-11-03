@echo off
setlocal
cd /d "%~dp0"
title BCI Studio - EEG self-test
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" bci_studio_gui.py %*
  goto finished
)
if exist "%USERPROFILE%\.bci_studio\venv\Scripts\python.exe" (
  "%USERPROFILE%\.bci_studio\venv\Scripts\python.exe" bci_studio_gui.py %*
  goto finished
)
where py >nul 2>&1
if errorlevel 1 (python bci_studio_gui.py %*) else (py -3 bci_studio_gui.py %*)
:finished
set "BCI_EXIT=%ERRORLEVEL%"
if errorlevel 1 echo BCI Studio stopped with an error. Run Setup_BCI_Studio.bat and check README_LAB_GUIDE.md.
pause
exit /b %BCI_EXIT%
