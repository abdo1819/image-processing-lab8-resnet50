# Lab 8 - Offline Setup  (Windows PowerShell)
# =============================================
# Run ONCE on each student machine after copying offline_packages/.
# No Python installation or admin rights required.
#
# Usage (from the try_2026 directory):
#   powershell -ExecutionPolicy Bypass -File setup\install_offline.ps1

$ErrorActionPreference = "Continue"   # external exe stderr must not kill the script

$RootDir   = Split-Path -Parent $PSScriptRoot
$EmbedDir  = Join-Path $RootDir "offline_packages\python_embedded"
$PipDir    = Join-Path $RootDir "offline_packages\pip_packages"
$PythonDir = Join-Path $RootDir "lab8_python"
$PythonExe = Join-Path $PythonDir "python.exe"
$SitePkgs  = Join-Path $PythonDir "Lib\site-packages"

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}

function Write-OK($msg)   { Write-Host "    OK  $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "    ... $msg" -ForegroundColor DarkGray }
function Write-Fail($msg) {
    Write-Host "    ERR $msg" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

function Elapsed($sw) { "({0:N0}s)" -f $sw.Elapsed.TotalSeconds }

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Lab 8 - Offline Setup" -ForegroundColor Cyan
Write-Host " Python 3.12 embeddable -- no install needed" -ForegroundColor DarkGray
Write-Host "============================================" -ForegroundColor Cyan

# ── Step 1: Extract Python embeddable ZIP ────────────────────────────────────
Write-Step 1 "Extract Python 3.12 embeddable package"

if (Test-Path $PythonExe) {
    $v = & $PythonExe --version 2>&1
    Write-OK "Already extracted: $v  (skipping)"
} else {
    $embedZip = Get-ChildItem -Path $EmbedDir -Filter "python-3.12*-embed-amd64.zip" `
                    -ErrorAction SilentlyContinue | Select-Object -First 1

    if ($null -eq $embedZip) { Write-Fail "Embeddable ZIP not found in: $EmbedDir" }

    Write-Info "Source : $($embedZip.Name)  ($([math]::Round($embedZip.Length/1MB,1)) MB)"
    Write-Info "Target : $PythonDir"
    Write-Info "Extracting files (this takes ~10-30 seconds)..."

    New-Item -ItemType Directory -Force $PythonDir | Out-Null

    # Use .NET ZipFile -- much faster than Expand-Archive
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $sw  = [System.Diagnostics.Stopwatch]::StartNew()
    $zip = [System.IO.Compression.ZipFile]::OpenRead($embedZip.FullName)
    $total = $zip.Entries.Count
    $i = 0
    Write-Host "    [" -NoNewline -ForegroundColor DarkGray
    foreach ($entry in $zip.Entries) {
        $i++
        $dest = Join-Path $PythonDir $entry.FullName
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Force $destDir | Out-Null }
        if ($entry.Name -ne "") {
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
        }
        # Print a dot every 5 files so the student sees progress
        if ($i % 5 -eq 0) { Write-Host "." -NoNewline -ForegroundColor DarkGray }
    }
    $zip.Dispose()
    Write-Host "]" -ForegroundColor DarkGray  # close progress bracket

    Write-OK "Extracted $total files  $(Elapsed $sw)"

    # Enable site-packages: add Lib\site-packages to python312._pth
    Write-Info "Configuring site-packages path..."
    $pthFile = Get-ChildItem -Path $PythonDir -Filter "python3*._pth" | Select-Object -First 1
    $lines   = Get-Content $pthFile.FullName
    if ($lines -notcontains "Lib\site-packages") {
        Add-Content $pthFile.FullName "`nLib\site-packages"
    }
    New-Item -ItemType Directory -Force $SitePkgs | Out-Null
    Write-OK "python312._pth configured"
}

# ── Step 2: Bootstrap pip ────────────────────────────────────────────────────
Write-Step 2 "Bootstrap pip into lab8_python"

# Probe silently -- "No module named pip" stderr must not crash the script
$null = & $PythonExe -m pip --version 2>&1
$pipPresent = ($LASTEXITCODE -eq 0)

if ($pipPresent) {
    $pipVer = (& $PythonExe -m pip --version 2>&1) -join ""
    Write-OK "pip already present: $pipVer  (skipping)"
} else {
    $pipWhl = Get-ChildItem -Path $PipDir -Filter "pip-*.whl" |
                  Sort-Object Name -Descending | Select-Object -First 1
    if ($null -eq $pipWhl) { Write-Fail "pip wheel not found in: $PipDir" }

    Write-Info "Extracting pip wheel directly to site-packages..."
    Write-Info "Source: $($pipWhl.Name)"

    # Embeddable Python ignores PYTHONPATH when a _pth file is present,
    # so we must extract the wheel (it is just a zip) into site-packages.
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $whlZip = [System.IO.Compression.ZipFile]::OpenRead($pipWhl.FullName)
    $extracted = 0
    foreach ($entry in $whlZip.Entries) {
        if ($entry.Name -eq "") { continue }  # skip directory entries
        $dest = Join-Path $SitePkgs $entry.FullName
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Force $destDir | Out-Null }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
        $extracted++
    }
    $whlZip.Dispose()
    Write-OK "pip extracted ($extracted files)  $(Elapsed $sw)"

    # Verify pip is now importable
    $null = & $PythonExe -m pip --version 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Fail "pip still not importable after extraction -- check site-packages path." }

    Write-Info "Installing setuptools + wheel via pip (needed by some packages)..."
    $sw2 = [System.Diagnostics.Stopwatch]::StartNew()
    & $PythonExe -m pip install `
        --no-index --find-links="$PipDir" `
        --target="$SitePkgs" `
        --quiet `
        setuptools wheel
    if ($LASTEXITCODE -ne 0) { Write-Fail "setuptools/wheel install failed." }
    Write-OK "setuptools + wheel installed  $(Elapsed $sw2)"
}


# ── Step 3: Install lab packages ─────────────────────────────────────────────
Write-Step 3 "Install lab packages (torch CPU + dependencies)"

if (-not (Test-Path $PipDir)) { Write-Fail "pip_packages folder not found: $PipDir" }

$packages = @("torch","torchvision","numpy","matplotlib","seaborn",
              "scikit-learn","Pillow","tqdm")

# Check which are already installed to give useful feedback
Write-Info "Checking already-installed packages..."
$installedRaw = & $PythonExe -m pip list --path "$SitePkgs" 2>&1
$missing = $packages | Where-Object {
    $pkg = $_
    -not ($installedRaw | Where-Object { $_ -match "(?i)^$([regex]::Escape($pkg))\s" })
}

if ($missing.Count -eq 0) {
    Write-OK "All packages already installed  (skipping)"
} else {
    Write-Info "Installing: $($missing -join ', ')"
    Write-Info "Source    : $PipDir"
    Write-Info "torch is ~110 MB -- extraction can take 5-10 min on a slow machine."
    Write-Info "The screen may appear frozen during torch install -- that is normal."
    Write-Info "DO NOT close this window.  Please wait..."
    Write-Host ""

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $PythonExe -m pip install `
        --no-index --find-links="$PipDir" `
        --target="$SitePkgs" `
        @packages   # show output so students see progress per-package

    if ($LASTEXITCODE -ne 0) { Write-Fail "Package installation failed." }
    Write-Host ""
    Write-OK "All packages installed  $(Elapsed $sw)"
}

# ── Step 4: Smoke test ───────────────────────────────────────────────────────
Write-Step 4 "Smoke test -- import torch and torchvision"
Write-Info "Running import check..."

$test = & $PythonExe -c @"
import torch, torchvision, numpy, matplotlib, seaborn, sklearn, PIL, tqdm
print('torch       ' + torch.__version__)
print('torchvision ' + torchvision.__version__)
print('numpy       ' + numpy.__version__)
print('device      ' + ('cuda' if torch.cuda.is_available() else 'cpu (expected)'))
"@ 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host $test -ForegroundColor Red
    Write-Fail "Import test failed -- see errors above."
}

$test | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
Write-OK "All imports succeeded"

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host " To start working, open PowerShell in the try_2026 folder and run:"
Write-Host "   . .\setup\activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host " Then run the lab scripts:"
Write-Host "   python inference.py    (Part 1)"
Write-Host "   python finetune.py     (Part 2)"
Write-Host ""
Read-Host "Press Enter to exit"

