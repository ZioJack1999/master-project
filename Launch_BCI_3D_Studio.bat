@echo off
title BCI 3D Live Studio (60 FPS)
echo ======================================================================
echo    STARTING BCI 3D LIVE STUDIO & REALISTIC VIRTUAL HAND (60 FPS)
echo ======================================================================
echo.
echo Launching BCI Studio Web Dashboard at http://127.0.0.1:8050 ...

REM Try Anaconda multi_nn environment first if it exists on this PC
if exist "C:\tools\Anaconda3\envs\multi_nn\python.exe" (
    "C:\tools\Anaconda3\envs\multi_nn\python.exe" bci_studio_gui.py
    pause
    exit /b
)

REM Otherwise use system python / active environment
python bci_studio_gui.py
if %errorlevel% neq 0 (
    echo.
    echo Trying python via py launcher...
    py -3 bci_studio_gui.py
)
pause
