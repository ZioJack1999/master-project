@echo off
setlocal
cd /d "%~dp0"
title BCI 3D Virtual Hand (Standalone Viewer)
echo ======================================================================
echo    STARTING BCI 3D VIRTUAL HAND (STANDALONE SECOND WINDOW)
echo ======================================================================
echo.
echo Opening 3D Virtual Hand in default browser...
if exist "virtual_hand.html" (
    start "" "http://127.0.0.1:8052/virtual_hand.html"
) else (
    echo Error: virtual_hand.html not found.
    pause
    exit /b 1
)
exit /b 0
