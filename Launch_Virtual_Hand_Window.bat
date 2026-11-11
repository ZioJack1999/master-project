@echo off
setlocal
title 3D Virtual Hand Window

rem Resolve repository directory
set "TARGET_DIR=%~dp0"
if exist "%TARGET_DIR%virtual_hand_viewer.py" goto found_dir

set "TARGET_DIR=%~dp0Academic & Projects\Master Project\master-project-debug\"
if exist "%TARGET_DIR%virtual_hand_viewer.py" goto found_dir

set "TARGET_DIR=%USERPROFILE%\Desktop\Academic & Projects\Master Project\master-project-debug\"
if exist "%TARGET_DIR%virtual_hand_viewer.py" goto found_dir

set "TARGET_DIR=C:\Users\giaco\Desktop\Academic & Projects\Master Project\master-project-debug\"
if exist "%TARGET_DIR%virtual_hand_viewer.py" goto found_dir

set "TARGET_DIR=%~dp0BCI_EEGMI_Results_and_Decoder\"
if exist "%TARGET_DIR%virtual_hand_viewer.py" goto found_dir

echo [ERROR] Could not find virtual_hand_viewer.py in target workspace.
pause
exit /b 1

:found_dir
cd /d "%TARGET_DIR%"

echo ======================================================================
echo       LAUNCHING 3D VIRTUAL HAND DESKTOP WINDOW
echo ======================================================================
echo Target Directory: "%TARGET_DIR%"
echo.

rem Select Python interpreter (.venv, Conda, or system python)
if exist "%TARGET_DIR%.venv\Scripts\python.exe" (
    set "PY_EXE=%TARGET_DIR%.venv\Scripts\python.exe"
    goto run_viewer
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PY_EXE=%USERPROFILE%\miniconda3\python.exe"
    goto run_viewer
)
if exist "C:\tools\Anaconda3\envs\multi_nn\python.exe" (
    set "PY_EXE=C:\tools\Anaconda3\envs\multi_nn\python.exe"
    goto run_viewer
)
if exist "%USERPROFILE%\.bci_studio\venv\Scripts\python.exe" (
    set "PY_EXE=%USERPROFILE%\.bci_studio\venv\Scripts\python.exe"
    goto run_viewer
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PY_EXE=python"
    goto run_viewer
)

echo [ERROR] Python environment was not found. Please install Python or set up .venv.
pause
exit /b 1

:run_viewer
"%PY_EXE%" virtual_hand_viewer.py %*
set "EXIT_CODE=%ERRORLEVEL%"
if errorlevel 1 (
    echo.
    echo 3D Virtual Hand Viewer exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
