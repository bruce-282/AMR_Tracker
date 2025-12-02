@echo off
chcp 65001 >nul
cd /d "C:\Users\junda\OneDrive\문서\cmes_repo\AMR_Tracker"
python "C:\Users\junda\OneDrive\문서\cmes_repo\AMR_Tracker\run_server.py" --host "0.0.0.0" --port 10000 --preset "camera_tracking"
pause

