@echo off
setlocal
cd /d "%~dp0"
set "THREECLASS_RUNTIME=%USERPROFILE%\.bci_threeclass\venv"
where py >nul 2>&1
if errorlevel 1 goto use_python
py -3.11 -c "import struct,sys; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, '64-bit Python 3.11 required'"
if errorlevel 1 goto failed
py -3.11 -m venv "%THREECLASS_RUNTIME%"
if errorlevel 1 goto failed
goto install
:use_python
python -c "import struct,sys; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, '64-bit Python 3.11 required'"
if errorlevel 1 goto failed
python -m venv "%THREECLASS_RUNTIME%"
if errorlevel 1 goto failed
:install
"%THREECLASS_RUNTIME%\Scripts\python.exe" -c "import struct,sys; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, '64-bit Python 3.11 required'"
if errorlevel 1 goto failed
"%THREECLASS_RUNTIME%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Three-class setup complete. Run Launch_THREECLASS_Lab.bat.
echo This environment is separate from the binary application.
pause
exit /b 0
:failed
echo Three-class setup failed. Install 64-bit Python 3.11 and check the error above.
echo Vendor acquisition and a compatible LSL bridge must be installed separately.
pause
exit /b 1
