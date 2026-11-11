@echo off
setlocal
if exist "%~dp0BCI_EEGMI_Results_and_Decoder\bci_studio_gui.py" (
    cd /d "%~dp0BCI_EEGMI_Results_and_Decoder"
) else if exist "%~dp0Academic & Projects\Master Project\master-project-debug\bci_studio_gui.py" (
    cd /d "%~dp0Academic & Projects\Master Project\master-project-debug"
) else (
    cd /d "%~dp0"
)
title Binary 3D BCI Demo (Rest vs Right-Hand Grasp)
echo ======================================================================
echo    STARTING BINARY 3D BCI DEMO (REST / RIGHT-HAND VOLAR GRASP)
echo ======================================================================
echo.
echo Launching Binary Studio Web Dashboard at http://127.0.0.1:8052 ...
echo Launching 3D Virtual Hand Window at http://127.0.0.1:8052/virtual_hand ...
echo.

if exist "C:\tools\Anaconda3\envs\multi_nn\python.exe" (
    "C:\tools\Anaconda3\envs\multi_nn\python.exe" bci_studio_gui.py --demo --port 8052 --open-virtual-hand %*
    goto finished
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" bci_studio_gui.py --demo --port 8052 --open-virtual-hand %*
    goto finished
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    "%USERPROFILE%\miniconda3\python.exe" bci_studio_gui.py --demo --port 8052 --open-virtual-hand %*
    goto finished
)
where py >nul 2>&1
if errorlevel 1 (python bci_studio_gui.py --demo --port 8052 --open-virtual-hand %*) else (py -3 bci_studio_gui.py --demo --port 8052 --open-virtual-hand %*)

:finished
set "BCI_EXIT=%ERRORLEVEL%"
pause
exit /b %BCI_EXIT%
