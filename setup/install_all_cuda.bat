@echo off
REM ============================================
REM Complete CUDA 12.8 + PyTorch Installation Script
REM ============================================

echo.
echo ============================================
echo Complete CUDA 12.8 + PyTorch Installation
echo ============================================
echo.
echo This script will install:
echo   0. Visual C++ Redistributable 2015-2022 (required for PyTorch)
echo   1. CUDA 12.8.0
echo   2. PyTorch with CUDA 12.8 support
echo   3. cuDNN (optional)
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul
echo.

REM Step 0: Install Visual C++ Redistributable
echo ============================================
echo Step 0/4: Installing Visual C++ Redistributable 2015-2022
echo ============================================
call install_vcredist.bat
if %errorLevel% neq 0 (
    echo [WARNING] Visual C++ Redistributable installation may have failed.
    echo Continuing anyway...
)
echo.
echo Press any key to continue to CUDA installation...
pause >nul
echo.

REM Step 1: Install CUDA 12.8
echo ============================================
echo Step 1/4: Installing CUDA 12.8
echo ============================================
call install_cuda_12.8.bat
if %errorLevel% neq 0 (
    echo [ERROR] CUDA installation failed. Aborting.
    pause
    exit /b 1
)

echo.
echo Press any key to continue to PyTorch installation...
pause >nul
echo.

REM Step 2: Install PyTorch
echo ============================================
echo Step 2/4: Installing PyTorch (CUDA 12.8)
echo ============================================
call install_pytorch_cuda128.bat
if %errorLevel% neq 0 (
    echo [ERROR] PyTorch installation failed. Aborting.
    pause
    exit /b 1
)

echo.
echo Press any key to continue to cuDNN installation (optional)...
pause >nul
echo.

REM Step 3: Install cuDNN (optional)
echo ============================================
echo Step 3/4: Installing cuDNN (Optional)
echo ============================================
echo Do you want to install cuDNN? (Y/N)
set /p install_cudnn="Enter choice: "

if /i "%install_cudnn%"=="Y" (
    call install_cudnn.bat
    if %errorLevel% neq 0 (
        echo [WARNING] cuDNN installation failed, but this is optional.
    )
) else (
    echo [INFO] Skipping cuDNN installation.
)

echo.
echo ============================================
echo All installations completed!
echo ============================================
echo.
echo Summary:
echo   - CUDA 12.8.0: Installed
echo   - PyTorch: Installed (CUDA 12.8 support)
if /i "%install_cudnn%"=="Y" (
    echo   - cuDNN: Installed
) else (
    echo   - cuDNN: Skipped
)
echo.
echo Please restart your terminal/command prompt before using PyTorch.
echo.
pause

