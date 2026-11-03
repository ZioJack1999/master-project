@echo off
setlocal
cd /d "%~dp0"
set "THREECLASS_PYTHON=%USERPROFILE%\.bci_threeclass\venv\Scripts\python.exe"
if exist "%THREECLASS_PYTHON%" goto launch
if exist ".venv\Scripts\python.exe" (
  set "THREECLASS_PYTHON=%CD%\.venv\Scripts\python.exe"
  goto launch
)
echo Run Setup_THREECLASS_Lab.bat first. The three-class runtime uses its own environment.
pause
exit /b 1
:launch
"%THREECLASS_PYTHON%" verify_threeclass_bundle.py
if errorlevel 1 (
  echo Three-class package integrity check failed. Review the listed files.
  pause
  exit /b 1
)
"%THREECLASS_PYTHON%" threeclass_studio.py %*
set "THREECLASS_EXIT=%ERRORLEVEL%"
if not "%THREECLASS_EXIT%"=="0" echo Three-class studio stopped with an error. Read README.md and check the message above.
pause
exit /b %THREECLASS_EXIT%
