# Lab 8 - Activate lab Python for the current PowerShell session
# ==============================================================
# Dot-source this script to add lab8_python to your PATH:
#   . .\setup\activate.ps1
#
# After activation, just type: python inference.py
# This only affects the current terminal window.

$PythonDir = Join-Path (Split-Path -Parent $PSScriptRoot) "lab8_python"
$PythonExe = Join-Path $PythonDir "python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: lab8_python\python.exe not found." -ForegroundColor Red
    Write-Host "Run setup first:" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy Bypass -File setup\install_offline.ps1" -ForegroundColor Yellow
    return
}

# Prepend lab8_python to PATH for this session only
$env:PATH      = "$PythonDir;$env:PATH"
$env:PYTHONPATH = ""  # clear any stale PYTHONPATH

$v = & $PythonExe --version 2>&1
Write-Host "Lab 8 Python activated: $v" -ForegroundColor Green
Write-Host "  Location : $PythonDir" -ForegroundColor DarkGray
Write-Host "  Run with : python inference.py   or   python finetune.py" -ForegroundColor DarkGray
