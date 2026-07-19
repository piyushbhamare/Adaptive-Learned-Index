@echo off
REM =============================================================================
REM run_all.bat — NLI Full Benchmark Suite
REM Windows CMD native — No Bash, No WSL, No MSYS2, No Cygwin required
REM
REM Requirements:
REM   - MinGW g++ in PATH  (e.g. from MSYS2: C:\msys64\mingw64\bin in PATH)
REM   - Python 3 in PATH
REM
REM Usage:
REM   run_all.bat [sosd_data_dir] [results_dir] [size1] [size2] [size3]
REM
REM Examples:
REM   run_all.bat sosd_data results
REM   run_all.bat sosd_data results 100000
REM   run_all.bat sosd_data results 100000 1000000
REM   run_all.bat sosd_data results 100000 1000000 200000000
REM
REM Defaults:
REM   sosd_data_dir = sosd_data
REM   results_dir   = results
REM   sizes         = 100000 1000000 200000000
REM =============================================================================

setlocal EnableDelayedExpansion

REM ── Script directory (so it works from any CWD) ────────────────────────────
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ── Arguments ────────────────────────────────────────────────────────────────
set "DATA_DIR=%~1"
set "OUT_DIR=%~2"
if "%DATA_DIR%"=="" set "DATA_DIR=sosd_data"
if "%OUT_DIR%"==""  set "OUT_DIR=results"

REM ── Key sizes from remaining args (or interactive menu) ──────────────────
set "SIZES_MAIN="
if not "%~3"=="" set "SIZES_MAIN=%~3"
if not "%~4"=="" set "SIZES_MAIN=%SIZES_MAIN% %~4"
if not "%~5"=="" set "SIZES_MAIN=%SIZES_MAIN% %~5"
if not "%~6"=="" set "SIZES_MAIN=%SIZES_MAIN% %~6"

if "%SIZES_MAIN%"=="" (
    echo.
    echo  +----------------------------------------------------+
    echo  ^|       Select key sizes to benchmark               ^|
    echo  ^|                                                   ^|
    echo  ^|  1^)  100K only         ~1-2 min,  low RAM         ^|
    echo  ^|  2^)  1M only           ~3-5 min,  moderate RAM    ^|
    echo  ^|  3^)  100K + 1M         ~5-7 min,  moderate RAM    ^|
    echo  ^|  4^)  100K+1M+200M  FULL - needs 8+ GB RAM         ^|
    echo  +----------------------------------------------------+
    echo.
    set /p "MENU_CHOICE=  Enter choice [1-4]: "
    if "!MENU_CHOICE!"=="1" set "SIZES_MAIN=100000"
    if "!MENU_CHOICE!"=="2" set "SIZES_MAIN=1000000"
    if "!MENU_CHOICE!"=="3" set "SIZES_MAIN=100000 1000000"
    if "!MENU_CHOICE!"=="4" set "SIZES_MAIN=100000 1000000 200000000"
    if "!SIZES_MAIN!"=="" (
        echo [ERROR] Invalid choice. Run again and enter 1, 2, 3, or 4.
        exit /b 1
    )
)

REM Drift sizes: 143000 (~100K model keys), 1000000 (700K), 1430000 (~1M)
set "SIZES_DRIFT=143000 1000000 1430000"

set "BUILD_DIR=build"
set "FIG_DIR=figures"

echo.
echo ============================================================
echo  NLI Full Benchmark Suite  [Windows / CMD]
echo ============================================================
echo  Data dir    : %DATA_DIR%
echo  Results dir : %OUT_DIR%
echo  Figures dir : %FIG_DIR%
echo  Key sizes   : !SIZES_MAIN!
echo  Drift sizes : !SIZES_DRIFT!
echo ============================================================
echo.

REM ── Check g++ ─────────────────────────────────────────────────────────────
where g++ >nul 2>&1
if errorlevel 1 (
    echo [ERROR] g++ not found in PATH.
    echo         Install MinGW via MSYS2: pacman -S mingw-w64-x86_64-gcc
    echo         Then add C:\msys64\mingw64\bin to your PATH.
    exit /b 1
)
for /f "tokens=*" %%v in ('g++ --version 2^>^&1 ^| findstr /i "g++"') do (
    echo [INFO]  Compiler: %%v
    goto :gxx_done
)
:gxx_done

REM ── Check Python ──────────────────────────────────────────────────────────
set "PYTHON="
for %%P in (python3 python) do (
    if "!PYTHON!"=="" (
        where %%P >nul 2>&1
        if not errorlevel 1 (
            for /f %%V in ('%%P -c "import sys; print(sys.version_info.major)" 2^>nul') do (
                if "%%V"=="3" set "PYTHON=%%P"
            )
        )
    )
)
if "!PYTHON!"=="" (
    echo [ERROR] Python 3 not found. Install from https://python.org and add to PATH.
    exit /b 1
)
for /f "tokens=*" %%v in ('!PYTHON! --version 2^>^&1') do echo [INFO]  Python:   %%v

REM ── Check datasets ────────────────────────────────────────────────────────
for %%D in (books_200M_uint64 fb_200M_uint64 wiki_ts_200M_uint64) do (
    if exist "%DATA_DIR%\%%D" (
        echo [INFO]  Dataset OK : %DATA_DIR%\%%D
    ) else (
        echo [WARN]  Dataset MISSING : %DATA_DIR%\%%D -- will be skipped
    )
)

REM ── Create directories ─────────────────────────────────────────────────────
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%OUT_DIR%"   mkdir "%OUT_DIR%"
if not exist "%FIG_DIR%"   mkdir "%FIG_DIR%"

REM ── Python dependencies ───────────────────────────────────────────────────
echo [INFO]  Installing Python dependencies...
!PYTHON! -m pip install --quiet --upgrade matplotlib pandas numpy 2>nul
if errorlevel 1 echo [WARN]  pip install reported an error - figures may fail. Run: pip install matplotlib pandas numpy

REM ── Node / docx dependency ────────────────────────────────────────────────
set "NODE_OK=0"
where node >nul 2>&1 && set "NODE_OK=1"
if "!NODE_OK!"=="1" (
    if not exist "node_modules\docx" (
        echo [INFO]  Installing docx npm package...
        npm install --silent docx >nul 2>&1
        if errorlevel 1 (
            echo [WARN]  npm install docx failed -- Word document will not be generated.
            set "NODE_OK=0"
        ) else (
            echo [INFO]  docx npm package ready.
        )
    )
) else (
    echo [WARN]  node.exe not found in PATH -- Word document will not be generated.
    echo         Install Node.js from https://nodejs.org to enable .docx output.
)

echo.
echo ============================================================
echo  Step 1/6 -- Compile
echo ============================================================
echo.

set "CXXFLAGS=-std=c++17 -O3 -march=native -mpopcnt -funroll-loops -ffast-math -I include"

REM Remove old Linux ELF binaries (no extension) that would shadow .exe
if exist "%BUILD_DIR%\nli_benchmark"  del /f "%BUILD_DIR%\nli_benchmark"
if exist "%BUILD_DIR%\nli_drift"      del /f "%BUILD_DIR%\nli_drift"
if exist "%BUILD_DIR%\nli_ablation"   del /f "%BUILD_DIR%\nli_ablation"

REM Compile nli_benchmark
echo [INFO]  Compiling nli_benchmark...
g++ %CXXFLAGS% benchmark\main_benchmark.cpp -o "%BUILD_DIR%\nli_benchmark.exe"
if errorlevel 1 (
    echo [ERROR] Compile failed: nli_benchmark
    exit /b 1
)
if not exist "%BUILD_DIR%\nli_benchmark.exe" (
    echo [ERROR] nli_benchmark.exe was not created despite no compile error.
    exit /b 1
)
echo [INFO]  nli_benchmark.exe OK

REM Compile nli_drift
echo [INFO]  Compiling nli_drift...
g++ %CXXFLAGS% benchmark\drift_benchmark.cpp -o "%BUILD_DIR%\nli_drift.exe"
if errorlevel 1 (
    echo [ERROR] Compile failed: nli_drift
    exit /b 1
)
if not exist "%BUILD_DIR%\nli_drift.exe" (
    echo [ERROR] nli_drift.exe was not created.
    exit /b 1
)
echo [INFO]  nli_drift.exe OK

REM Compile nli_ablation
echo [INFO]  Compiling nli_ablation...
g++ %CXXFLAGS% benchmark\ablation_benchmark.cpp -o "%BUILD_DIR%\nli_ablation.exe"
if errorlevel 1 (
    echo [ERROR] Compile failed: nli_ablation
    exit /b 1
)
if not exist "%BUILD_DIR%\nli_ablation.exe" (
    echo [ERROR] nli_ablation.exe was not created.
    exit /b 1
)
echo [INFO]  nli_ablation.exe OK

echo.
echo ============================================================
echo  Step 2/6 -- Main Benchmark
echo ============================================================
echo  Indexes: B-Tree / ALEX / PGM / RMI / NLI-Full / NLI-Linear / NLI-NoDrift
echo  Sizes:   %SIZES_MAIN%
echo.

"%BUILD_DIR%\nli_benchmark.exe" "%DATA_DIR%" "%OUT_DIR%" %SIZES_MAIN%
if errorlevel 1 (
    echo [ERROR] nli_benchmark.exe failed.
    exit /b 1
)
echo [INFO]  Saved --^> %OUT_DIR%\benchmark_results.csv

echo.
echo ============================================================
echo  Step 3/6 -- Drift Detection Benchmark
echo ============================================================
echo  Sizes:   %SIZES_DRIFT%
echo.

"%BUILD_DIR%\nli_drift.exe" "%DATA_DIR%" "%OUT_DIR%" %SIZES_DRIFT%
if errorlevel 1 (
    echo [ERROR] nli_drift.exe failed.
    exit /b 1
)
echo [INFO]  Saved --^> %OUT_DIR%\drift_results.csv

echo.
echo ============================================================
echo  Step 4/6 -- Ablation Study
echo ============================================================
echo  Sizes:   %SIZES_MAIN%
echo.

"%BUILD_DIR%\nli_ablation.exe" "%DATA_DIR%" "%OUT_DIR%" %SIZES_MAIN%
if errorlevel 1 (
    echo [ERROR] nli_ablation.exe failed.
    exit /b 1
)
echo [INFO]  Saved --^> %OUT_DIR%\ablation_results.csv

echo.
echo ============================================================
echo  Step 5/6 -- Figures ^& Markdown Report
echo ============================================================
echo.

!PYTHON! generate_figures.py "!OUT_DIR!" "!FIG_DIR!"
if errorlevel 1 (
    echo [WARN]  generate_figures.py failed -- check Python/matplotlib install.
) else (
    echo [INFO]  Figures and markdown report generated.
)

echo.
echo ============================================================
echo  Step 6/6 -- Word Document  (NLI_v7.1_Benchmark_Results.docx)
echo ============================================================
echo.

if "!NODE_OK!"=="1" (
    node make_report.js "!OUT_DIR!" "!FIG_DIR!"
    if errorlevel 1 (
        echo [WARN]  make_report.js failed -- check node / docx install.
    ) else (
        echo [INFO]  NLI_v7.1_Benchmark_Results.docx generated.
    )
) else (
    echo [SKIP]  node not available -- skipping Word document generation.
)

echo.
echo ============================================================
echo  Done!
echo ============================================================
echo.
echo  Results:
if exist "!OUT_DIR!\benchmark_results.csv"        (echo    benchmark_results.csv             OK) else (echo    benchmark_results.csv             MISSING)
if exist "!OUT_DIR!\drift_results.csv"             (echo    drift_results.csv                 OK) else (echo    drift_results.csv                 MISSING)
if exist "!OUT_DIR!\ablation_results.csv"          (echo    ablation_results.csv              OK) else (echo    ablation_results.csv              MISSING)
if exist "!OUT_DIR!\benchmark_report.md"           (echo    benchmark_report.md               OK) else (echo    benchmark_report.md               MISSING)
if exist "NLI_v7.1_Benchmark_Results.docx"         (echo    NLI_v7.1_Benchmark_Results.docx   OK) else (echo    NLI_v7.1_Benchmark_Results.docx   MISSING ^(node required^))
echo.
echo  Open NLI_v7.1_Benchmark_Results.docx for the full report.
echo.

endlocal
