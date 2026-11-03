@echo off
rem Metasoznanie TEST MODE: cycles back to back, separate archive_test, test printer, protocol sheets, debug screen.
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Run first:  powershell -ExecutionPolicy Bypass -File install\install.ps1
  pause
  exit /b 1
)
start "Metasoznanie service" /min ".venv\Scripts\python.exe" apophenia.py run --config config.test.yaml
timeout /t 15 /nobreak >nul
set EDGE="C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not exist %EDGE% set EDGE="C:\Program Files\Microsoft\Edge\Application\msedge.exe"
if not exist %EDGE% (
  echo Microsoft Edge not found. Open http://127.0.0.1:8765/ in any browser and press F11.
  exit /b 0
)
rem wait until the service page answers (up to 90 s), so Edge does not open an error page
powershell -NoProfile -Command "$t=0; while($t -lt 45){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',8765); $c.Close(); break } catch { Start-Sleep 2; $t++ } }"
rem separate profile: always a fresh Edge instance that honours the URL, no first-run pages
start "" %EDGE% --user-data-dir="%LOCALAPPDATA%\Metasoznanie\edge" --no-first-run --no-default-browser-check --disable-session-crashed-bubble --disable-features=TranslateUI,msEdgeStartupBoost --kiosk http://127.0.0.1:8765/ --edge-kiosk-type=fullscreen
