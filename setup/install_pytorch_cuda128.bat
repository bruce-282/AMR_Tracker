@echo off
REM ============================================
REM PyTorch Installation Script for CUDA 12.8
REM ============================================

echo.
echo ============================================
echo PyTorch Installation (CUDA 12.8)
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo [INFO] Python version:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] pip is not installed!
    echo Please install pip first.
    pause
    exit /b 1
)

echo [INFO] pip version:
pip --version
echo.

REM Add CUDA to PATH if installed but not in PATH
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
if exist "%CUDA_PATH%\bin\nvcc.exe" (
    set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
    echo [INFO] CUDA 12.8 found, added to PATH
    echo [INFO] CUDA version:
    nvcc --version
    echo.
) else (
    echo [WARNING] CUDA 12.8 not found at default location.
    echo This might be okay if CUDA is installed elsewhere.
    echo Continuing with PyTorch installation...
    echo.
)

echo [INFO] Installing PyTorch with CUDA 12.8 support...
echo This may take several minutes...
echo.

REM Add CUDA to PATH before installation
if exist "%CUDA_PATH%\bin\nvcc.exe" (
    set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%CUDA_PATH%\lib\x64;%PATH%"
    set "CUDA_HOME=%CUDA_PATH%"
)

REM Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if %errorLevel% neq 0 (
    echo [ERROR] PyTorch installation failed!
    pause
    exit /b 1
)

echo.
echo [INFO] Verifying PyTorch installation...
REM Ensure CUDA PATH is set for verification
if exist "%CUDA_PATH%\bin\nvcc.exe" (
    set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%CUDA_PATH%\lib\x64;%PATH%"
    REM Set CUDA environment variables for PyTorch
    set "CUDA_PATH=%CUDA_PATH%"
    set "CUDA_HOME=%CUDA_PATH%"
)
REM Use verify_pytorch.py script instead of inline Python for better error handling
python setup\verify_pytorch.py

if %errorLevel% neq 0 (
    echo [ERROR] PyTorch verification failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installation completed!
echo ============================================
echo.
echo PyTorch with CUDA 12.8 support has been installed.
echo.
pause

