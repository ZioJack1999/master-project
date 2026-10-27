@echo off
title BCI Studio 3D - Live Neuro-Engineering Console
echo ======================================================================
echo    STARTING BCI STUDIO 3D - LIVE NEURO-ENGINEERING CONSOLE (60 FPS)
echo ======================================================================
echo.
echo Launching BCI Studio Web Dashboard at http://127.0.0.1:8050 ...
python bci_studio_gui.py
if %errorlevel% neq 0 (
    echo.
    echo If 'python' failed, trying python via py launcher...
    py -3 bci_studio_gui.py
)
pause
