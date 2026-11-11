@echo off
setlocal
if exist "%~dp0BCI_EEGMI_Results_and_Decoder\bci_studio_gui.py" (
    cd /d "%~dp0BCI_EEGMI_Results_and_Decoder"
) else (
    cd /d "%~dp0"
)
title BCI 3D Live Studio (Unified Baseline)
echo ======================================================================
echo    STARTING BCI 3D LIVE STUDIO (UNIFIED BASELINE AND VIRTUAL HAND UDP 5005)
echo ======================================================================
echo.
echo Launching BCI Studio Web Dashboard at http://127.0.0.1:8052 ...

if exist "C:\tools\Anaconda3\envs\multi_nn\python.exe" (
    "C:\tools\Anaconda3\envs\multi_nn\python.exe" bci_studio_gui.py %*
    goto finished
)
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
if errorlevel 1 echo BCI Studio stopped with an error. Run Setup_BCI_Studio.bat.
pause
exit /b %BCI_EXIT%
