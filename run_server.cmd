@echo off

powershell -Command "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -like '*run_server.py*'} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force}"

if %errorlevel% == 0 (
    echo Server stopped successfully.
) else (
    echo No running server found or error occurred.
)
chcp 65001 >nul
cd /d "C:\Users\user\Documents\cmes_repo\AMR_Tracker"
python "C:\Users\user\Documents\cmes_repo\AMR_Tracker\run_server.py" --host "0.0.0.0" --port 10000 --preset "camera_tracking"
pause

