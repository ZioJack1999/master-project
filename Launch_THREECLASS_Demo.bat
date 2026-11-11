@echo off
setlocal
if exist "%~dp0BCI_EEGMI_Results_and_Decoder\threeclass_studio.py" (
    cd /d "%~dp0BCI_EEGMI_Results_and_Decoder"
) else if exist "%~dp0Academic & Projects\Master Project\master-project-debug\threeclass_studio.py" (
    cd /d "%~dp0Academic & Projects\Master Project\master-project-debug"
) else (
    cd /d "%~dp0"
)
title Three-Class 3D BCI Demo (Rest vs Grasp vs Extension)
echo ======================================================================
echo    STARTING THREE-CLASS 3D BCI DEMO (REST / GRASP / FINGER EXTENSION)
echo ======================================================================
echo.
echo Launching Three-Class Studio Web Dashboard at http://127.0.0.1:8051 ...
echo Launching 3D Virtual Hand Window at http://127.0.0.1:8051/virtual_hand ...
echo.

if exist "C:\tools\Anaconda3\envs\multi_nn\python.exe" (
    "C:\tools\Anaconda3\envs\multi_nn\python.exe" threeclass_studio.py --demo --port 8051 --open-virtual-hand %*
    goto finished
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" threeclass_studio.py --demo --port 8051 --open-virtual-hand %*
    goto finished
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    "%USERPROFILE%\miniconda3\python.exe" threeclass_studio.py --demo --port 8051 --open-virtual-hand %*
    goto finished
)
where py >nul 2>&1
if errorlevel 1 (python threeclass_studio.py --demo --port 8051 --open-virtual-hand %*) else (py -3 threeclass_studio.py --demo --port 8051 --open-virtual-hand %*)

:finished
set "BCI_EXIT=%ERRORLEVEL%"
pause
exit /b %BCI_EXIT%
