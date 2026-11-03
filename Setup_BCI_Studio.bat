@echo off
setlocal
cd /d "%~dp0"
set "BCI_RUNTIME=%USERPROFILE%\.bci_studio\venv"
where py >nul 2>&1
if errorlevel 1 (python -m venv "%BCI_RUNTIME%") else (py -3 -m venv "%BCI_RUNTIME%")
if errorlevel 1 goto failed
"%BCI_RUNTIME%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Run Launch_BCI_3D_Studio.bat.
pause
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.11 or 3.12, then retry. Check the error above.
pause
exit /b 1
