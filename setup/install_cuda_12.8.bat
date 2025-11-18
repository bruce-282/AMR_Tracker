@echo off
REM ============================================
REM CUDA 12.8 Installation Script for Windows
REM ============================================

echo.
echo ============================================
echo CUDA 12.8 Installation Script
echo ============================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script must be run as Administrator!
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo [INFO] Checking CUDA installation...
nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] NVIDIA GPU driver not found!
    echo Please install NVIDIA GPU driver first.
    pause
    exit /b 1
)

echo [INFO] NVIDIA GPU driver detected.
echo.

REM CUDA 12.8 download URL (official NVIDIA)
set CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe
set CUDA_INSTALLER=cuda_12.8.0_windows.exe

echo [INFO] Downloading CUDA 12.8.0 installer...
echo URL: %CUDA_URL%
echo.

REM Download CUDA installer
curl -L -o "%CUDA_INSTALLER%" "%CUDA_URL%"
if %errorLevel% neq 0 (
    echo [ERROR] Failed to download CUDA installer!
    echo Please download manually from:
    echo https://developer.nvidia.com/cuda-12-8-0-download-archive
    pause
    exit /b 1
)

echo [INFO] CUDA installer downloaded successfully.
echo.
echo [INFO] Starting CUDA 12.8.0 installation...
echo Please follow the installation wizard.
echo.
echo IMPORTANT: During installation, make sure to:
echo   1. Select "Express" installation (recommended)
echo   2. Keep default installation path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
echo   3. Wait for installation to complete
echo.

start /wait "" "%CUDA_INSTALLER%"

if %errorLevel% neq 0 (
    echo [ERROR] CUDA installation failed!
    pause
    exit /b 1
)

echo.
echo [INFO] Verifying CUDA installation...
nvcc --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] CUDA compiler (nvcc) not found in PATH.
    echo You may need to restart your terminal or add CUDA to PATH manually.
    echo Default CUDA path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
) else (
    echo [SUCCESS] CUDA 12.8 installed successfully!
    nvcc --version
)

echo.
echo [INFO] Cleaning up installer...
if exist "%CUDA_INSTALLER%" del "%CUDA_INSTALLER%"

echo.
echo ============================================
echo Installation completed!
echo ============================================
echo.
echo Next steps:
echo   1. Restart your terminal/command prompt
echo   2. Run: install_pytorch_cuda128.bat
echo   3. (Optional) Run: install_cudnn.bat
echo.
pause

