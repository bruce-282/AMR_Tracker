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

REM Check if CUDA is available
nvcc --version >nul 2>&1
if %errorLevel% neq 0 (
    echo This might be okay if CUDA is installed but PATH is not set.
    echo Continuing with PyTorch installation...
    echo.
) else (
    echo [INFO] CUDA version:
    nvcc --version
    echo.
)

echo [INFO] Installing PyTorch with CUDA 12.8 support...
echo This may take several minutes...
echo.

REM Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if %errorLevel% neq 0 (
    echo [ERROR] PyTorch installation failed!
    pause
    exit /b 1
)

echo.
echo [INFO] Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

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

