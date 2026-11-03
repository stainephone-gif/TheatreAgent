@echo off
rem Metasoznanie: stop the kiosk screen and the service (python.exe or pythonw.exe running apophenia.py).
taskkill /f /im msedge.exe >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Metasoznanie service" >nul 2>&1
wmic process where "commandline like '%%apophenia.py run%%'" call terminate >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*apophenia.py run*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
echo Stopped.
timeout /t 2 >nul
