# Регистрирует две задачи планировщика: служба (run.bat) и экран (kiosk.bat) при входе пользователя в систему,
# с перезапуском при сбое. Запускать от имени того пользователя, под которым настроен автовход.
#   powershell -ExecutionPolicy Bypass -File install\register_task.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$user = "$env:USERDOMAIN\$env:USERNAME"

$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit (New-TimeSpan -Days 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$trigger.Delay = "PT15S"

$a1 = New-ScheduledTaskAction -Execute "$root\install\run.bat" -WorkingDirectory $root
Register-ScheduledTask -TaskName "Metasoznanie-Service" -Action $a1 -Trigger $trigger -Settings $settings -User $user -RunLevel Limited -Force | Out-Null

$a2 = New-ScheduledTaskAction -Execute "$root\install\kiosk.bat" -WorkingDirectory $root
Register-ScheduledTask -TaskName "Metasoznanie-Screen" -Action $a2 -Trigger $trigger -Settings $settings -User $user -RunLevel Limited -Force | Out-Null

Write-Host "Задачи зарегистрированы: Metasoznanie-Service, Metasoznanie-Screen (запуск при входе $user)."
Write-Host "Проверить: Планировщик заданий → Библиотека. Удалить: Unregister-ScheduledTask -TaskName Metasoznanie-Service"
