# =============================================================================
# run_all.ps1 — NLI v5.0 Full Benchmark Suite
# PowerShell 5.1+ / PowerShell 7+  (Windows 11 native)
# No Bash, No WSL, No MSYS2, No Cygwin required.
#
# Requirements:
#   - MinGW g++ in PATH  (e.g. from MSYS2: C:\msys64\mingw64\bin)
#   - Python 3 in PATH
#
# Usage:
#   .\run_all.ps1 [sosd_data_dir] [results_dir] [size1] [size2] ...
#
# Examples:
#   .\run_all.ps1 sosd_data results
#   .\run_all.ps1 sosd_data results 100000
#   .\run_all.ps1 sosd_data results 100000 1000000
#   .\run_all.ps1 sosd_data results 100000 1000000 200000000
#
# If PowerShell blocks execution:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# =============================================================================

param(
    [string]$DataDir = "sosd_data",
    [string]$OutDir  = "results",
    [int[]] $Sizes   = @()          # empty = show interactive menu
)

$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Info  { param([string]$m) Write-Host "[INFO]  $m" -ForegroundColor Green }
function Warn  { param([string]$m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Err   { param([string]$m) Write-Host "[ERROR] $m" -ForegroundColor Red; exit 1 }
function Step  { param([string]$m) Write-Host "`n$('=' * 60)`n  $m`n$('=' * 60)" -ForegroundColor Cyan }

# ── Change to script directory ─────────────────────────────────────────────────
Set-Location -Path $PSScriptRoot

$BuildDir = "build"
$FigDir   = "figures"

# ── Interactive size selection (only when -Sizes not passed) ──────────────────
if ($Sizes.Count -eq 0) {
    Write-Host ""
    Write-Host "  +----------------------------------------------------+" -ForegroundColor Cyan
    Write-Host "  |        Select key sizes to benchmark               |" -ForegroundColor Cyan
    Write-Host "  |                                                    |" -ForegroundColor Cyan
    Write-Host "  |  1)  100K only         ~1-2 min,  low RAM          |" -ForegroundColor Cyan
    Write-Host "  |  2)  1M only           ~3-5 min,  moderate RAM     |" -ForegroundColor Cyan
    Write-Host "  |  3)  100K + 1M         ~5-7 min,  moderate RAM     |" -ForegroundColor Cyan
    Write-Host "  |  4)  100K + 1M + 200M  FULL - needs 8+ GB RAM      |" -ForegroundColor Cyan
    Write-Host "  +----------------------------------------------------+" -ForegroundColor Cyan
    Write-Host ""
    $choice = Read-Host "  Enter choice [1-4]"
    switch ($choice) {
        "1" { $Sizes = @(100000) }
        "2" { $Sizes = @(1000000) }
        "3" { $Sizes = @(100000, 1000000) }
        "4" { $Sizes = @(100000, 1000000, 200000000) }
        default { Err "Invalid choice '$choice'. Run again and enter 1, 2, 3, or 4." }
    }
}

# Drift sizes: 1M and above
$DriftSizes = $Sizes | Where-Object { $_ -ge 1000000 }
if ($DriftSizes.Count -eq 0) { $DriftSizes = @(1000000) }

Write-Host "`n$('=' * 60)" -ForegroundColor Cyan
Write-Host "  NLI v5.0 Full Benchmark Suite  [Windows / PowerShell]" -ForegroundColor Cyan
Write-Host "$('=' * 60)" -ForegroundColor Cyan
Info "Data dir   : $DataDir"
Info "Results dir: $OutDir"
Info "Figures dir: $FigDir"
Info "Key sizes  : $($Sizes -join ', ')"
Info "Drift sizes: $($DriftSizes -join ', ')"

# ── Sanity checks ─────────────────────────────────────────────────────────────
$gpp = Get-Command g++ -ErrorAction SilentlyContinue
if (-not $gpp) {
    Err "g++ not found in PATH.`n  Install MinGW via MSYS2: pacman -S mingw-w64-x86_64-gcc`n  Then add C:\msys64\mingw64\bin to PATH."
}
Info "Compiler: $(g++ --version 2>&1 | Select-Object -First 1)"

# Find Python 3
$Python = $null
foreach ($py in @("python3","python")) {
    $cmd = Get-Command $py -ErrorAction SilentlyContinue
    if ($cmd) {
        $ver = & $py -c "import sys; print(sys.version_info.major)" 2>$null
        if ($ver -eq "3") { $Python = $py; break }
    }
}
if (-not $Python) { Err "Python 3 not found. Install from https://python.org and add to PATH." }
Info "Python: $Python  $(& $Python --version 2>&1)"

# Datasets
foreach ($ds in @("books_200M_uint64","fb_200M_uint64","wiki_ts_200M_uint64")) {
    if (Test-Path "$DataDir\$ds") {
        $sz = (Get-Item "$DataDir\$ds").Length
        Info "Dataset OK : $DataDir\$ds  ($([math]::Round($sz/1MB)) MB)"
    } else {
        Warn "Dataset MISSING: $DataDir\$ds — will be skipped"
    }
}

# ── Create dirs ────────────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir   | Out-Null
New-Item -ItemType Directory -Force -Path $FigDir   | Out-Null

# ── Python packages ───────────────────────────────────────────────────────────
Info "Checking Python dependencies..."
& $Python -m pip install --quiet --upgrade matplotlib pandas numpy 2>$null
if ($LASTEXITCODE -ne 0) { Warn "pip install reported an error. Run: pip install matplotlib pandas numpy" }

# ── Compile ───────────────────────────────────────────────────────────────────
Step "Step 1/4 -- Compile"

$CxxFlags = "-std=c++17 -O3 -march=native -mpopcnt -funroll-loops -ffast-math -I include"

function Compile-Binary {
    param([string]$Src, [string]$OutName, [string]$Label)
    Info "Compiling $Label..."
    # Remove any stale Linux ELF that shadows the .exe
    $elfPath = Join-Path $BuildDir $OutName
    if (Test-Path $elfPath) { Remove-Item $elfPath -Force }
    $exePath = Join-Path $BuildDir "${OutName}.exe"
    # Run compiler — capture combined stdout+stderr
    $output = & g++ $CxxFlags.Split(' ') $Src -o $exePath 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host $output
        Err "Compile FAILED: $Label"
    }
    if (-not (Test-Path $exePath)) {
        Err "$exePath not created (compile succeeded but file missing)."
    }
    Info "$Label OK -> $exePath"
    return $exePath
}

$BinBench = Compile-Binary "benchmark\main_benchmark.cpp"     "nli_benchmark" "nli_benchmark"
$BinDrift = Compile-Binary "benchmark\drift_benchmark.cpp"    "nli_drift"     "nli_drift"
$BinAbl   = Compile-Binary "benchmark\ablation_benchmark.cpp" "nli_ablation"  "nli_ablation"

# ── Main benchmark ─────────────────────────────────────────────────────────────
Step "Step 2/4 -- Main Benchmark"
Info "Indexes: B-Tree / ALEX / PGM / RMI / NLI-Full / NLI-Linear / NLI-NoDrift"
Info "Sizes:   $($Sizes -join ' ')"
Write-Host ""

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $BinBench $DataDir $OutDir $Sizes
if ($LASTEXITCODE -ne 0) { Err "nli_benchmark.exe failed." }
$sw.Stop()
Info "Main benchmark done in $($sw.Elapsed.ToString('mm\:ss\.fff'))  ->  $OutDir\benchmark_results.csv"

# ── Drift benchmark ────────────────────────────────────────────────────────────
Step "Step 3/4 -- Drift Detection Benchmark"
Info "Sizes: $($DriftSizes -join ' ')"
Write-Host ""

$sw.Restart()
& $BinDrift $DataDir $OutDir $DriftSizes
if ($LASTEXITCODE -ne 0) { Err "nli_drift.exe failed." }
$sw.Stop()
Info "Drift benchmark done in $($sw.Elapsed.ToString('mm\:ss\.fff'))  ->  $OutDir\drift_results.csv"

# ── Ablation ───────────────────────────────────────────────────────────────────
Step "Step 4/4 -- Ablation Study"
Info "Sizes: $($Sizes -join ' ')"
Write-Host ""

$sw.Restart()
& $BinAbl $DataDir $OutDir $Sizes
if ($LASTEXITCODE -ne 0) { Err "nli_ablation.exe failed." }
$sw.Stop()
Info "Ablation done in $($sw.Elapsed.ToString('mm\:ss\.fff'))  ->  $OutDir\ablation_results.csv"

# ── Figures + report ───────────────────────────────────────────────────────────
Step "Step 5 -- Figures & Report"

& $Python generate_figures.py $OutDir $FigDir
if ($LASTEXITCODE -ne 0) { Warn "generate_figures.py failed -- check Python/matplotlib install." }
else { Info "Figures and report generated successfully." }

# ── Summary ────────────────────────────────────────────────────────────────────
Step "Complete!"
$resultFiles = @(
    "$OutDir\benchmark_results.csv",
    "$OutDir\drift_results.csv",
    "$OutDir\ablation_results.csv",
    "$OutDir\benchmark_report.md"
)
Write-Host "`n  Results:" -ForegroundColor Cyan
foreach ($f in $resultFiles) {
    if (Test-Path $f) { Write-Host "    $f  OK" -ForegroundColor Green }
    else              { Write-Host "    $f  MISSING" -ForegroundColor Red }
}
Write-Host "`n  Figures ($FigDir\):" -ForegroundColor Cyan
Get-ChildItem "$FigDir\*.png" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "    $($_.Name)"
}
Write-Host ""
Info "Open $OutDir\benchmark_report.md for the full report."
