@echo off
rem Screen: Microsoft Edge in kiosk mode on the local service page. Port must match display.port in config.yaml.
timeout /t 20 /nobreak >nul
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
