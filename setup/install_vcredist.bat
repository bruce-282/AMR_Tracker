@echo off
REM ============================================
REM Visual C++ Redistributable 2015-2022 Installation
REM ============================================

echo.
echo ============================================
echo Visual C++ Redistributable 2015-2022 Installer
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

REM Download URL for Visual C++ Redistributable 2015-2022
set VCREDIST_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe
set VCREDIST_INSTALLER=vc_redist.x64.exe

echo [INFO] Downloading Visual C++ Redistributable 2015-2022...
echo URL: %VCREDIST_URL%
echo.

REM Download installer
curl -L -o "%VCREDIST_INSTALLER%" "%VCREDIST_URL%"
if %errorLevel% neq 0 (
    echo [ERROR] Failed to download Visual C++ Redistributable!
    echo Please download manually from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    pause
    exit /b 1
)

echo [INFO] Visual C++ Redistributable downloaded successfully.
echo.
echo [INFO] Starting installation...
echo Please follow the installation wizard.
echo.

REM Install silently
"%VCREDIST_INSTALLER%" /install /quiet /norestart

if %errorLevel% neq 0 (
    echo [WARNING] Installation may have failed or already installed.
    echo This is okay if Visual C++ Redistributable is already installed.
)

echo.
echo [INFO] Cleaning up installer...
if exist "%VCREDIST_INSTALLER%" del "%VCREDIST_INSTALLER%"

echo.
echo ============================================
echo Installation completed!
echo ============================================
echo.
echo IMPORTANT: Please restart your terminal/command prompt after installation.
echo.
pause

