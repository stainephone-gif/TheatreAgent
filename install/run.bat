@echo off
rem Metasoznanie service. Started by Task Scheduler at logon; the scheduled task restarts it if it exits.
cd /d "%~dp0\.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
rem pythonw.exe: no console window; the log is in logs\apophenia.log
".venv\Scripts\pythonw.exe" apophenia.py run
