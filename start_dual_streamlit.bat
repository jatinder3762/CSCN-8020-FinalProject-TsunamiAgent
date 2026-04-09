@echo off
setlocal
cd /d "%~dp0"
python start_dual_dashboards.py %*
endlocal
