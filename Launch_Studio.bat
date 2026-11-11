@echo off
setlocal
cd /d "%~dp0" || exit /b 1

if defined BCI_STUDIO_PYTHON goto run
for %%P in (
    "C:\tools\Anaconda3\envs\multi_nn\python.exe"
    ".venv\Scripts\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\.bci_studio\venv\Scripts\python.exe"
) do if exist "%%~P" (
    set "BCI_STUDIO_PYTHON=%%~P"
    goto run
)

where py >nul 2>&1
if errorlevel 1 (
    python launch_studio.py %*
) else (
    py -3 launch_studio.py %*
)
goto finished

:run
"%BCI_STUDIO_PYTHON%" launch_studio.py %*

:finished
set "BCI_EXIT=%ERRORLEVEL%"
if not "%BCI_EXIT%"=="0" if "%~1"=="" pause
exit /b %BCI_EXIT%
