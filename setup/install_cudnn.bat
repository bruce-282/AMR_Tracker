@echo off
REM ============================================
REM cuDNN Installation Script for CUDA 12.x
REM ============================================

echo.
echo ============================================
echo cuDNN Installation (CUDA 12.x)
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

echo [INFO] Installing NVIDIA cuDNN for CUDA 12...
echo This package provides cuDNN libraries for PyTorch.
echo.

REM Install cuDNN
pip install nvidia-cudnn-cu12

if %errorLevel% neq 0 (
    echo [ERROR] cuDNN installation failed!
    pause
    exit /b 1
)

echo.
echo [INFO] Verifying cuDNN installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'cuDNN enabled: {torch.backends.cudnn.enabled if torch.cuda.is_available() else False}')"

if %errorLevel% neq 0 (
    echo [WARNING] cuDNN verification failed, but installation may have succeeded.
    echo Please verify manually.
)

echo.
echo ============================================
echo Installation completed!
echo ============================================
echo.
echo cuDNN has been installed.
echo.
pause

