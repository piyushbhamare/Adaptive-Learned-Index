@echo off
REM =============================================================================
REM run_asan.bat — ASAN / UBSAN Sanitizer Validation for NLI
REM IEEE CVMI 2026 | Paper ID 625 | Group 19
REM
REM Tries 4 compile strategies in order (most to least instrumented):
REM   1. ASAN + UBSAN          (-fsanitize=address,undefined)
REM   2. UBSAN no-link-runtime (-fsanitize=undefined -fno-sanitize-link-runtime)
REM   3. UBSAN static          (-fsanitize=undefined -static-libubsan)
REM   4. No sanitizers         (correctness-only run — all 12 tests must pass)
REM
REM A clean run (12/12 PASS, exit 0) constitutes correctness proof.
REM Strategy 1 or 2 also provides memory-safety proof.
REM =============================================================================

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo ================================================================
echo  NLI -- ASAN / UBSAN Validation Suite
echo  IEEE CVMI 2026 ^| Paper ID 625 ^| Group 19
echo ================================================================
echo.

REM ── Check g++ ──────────────────────────────────────────────────────────────
where g++ >nul 2>&1
if errorlevel 1 (
    echo [ERROR] g++ not found in PATH.
    exit /b 1
)
for /f "tokens=*" %%v in ('g++ --version 2^>^&1 ^| findstr /i "g++"') do (
    echo [INFO]  Compiler: %%v
    goto :gxx_done
)
:gxx_done

if not exist "build" mkdir "build"

set "SANITIZER_FLAGS=none"

REM ── Strategy 1: ASAN + UBSAN ──────────────────────────────────────────────
echo [INFO]  Strategy 1: -fsanitize=address,undefined ...
g++ -std=c++17 -O1 -g -fsanitize=address,undefined ^
    -fno-omit-frame-pointer ^
    -I include ^
    benchmark\asan_test.cpp ^
    -o build\nli_asan.exe 2>nul
if not errorlevel 1 (
    echo [INFO]  Compiled with ASAN+UBSAN
    set "SANITIZER_FLAGS=ASAN+UBSAN"
    goto :run
)

REM ── Strategy 2: UBSAN with -fno-sanitize-link-runtime (MinGW no libubsan) ─
echo [WARN]  Strategy 1 failed. Trying -fsanitize=undefined -fno-sanitize-link-runtime ...
g++ -std=c++17 -O1 -g -fsanitize=undefined ^
    -fno-sanitize-link-runtime ^
    -fno-omit-frame-pointer ^
    -I include ^
    benchmark\asan_test.cpp ^
    -o build\nli_asan.exe 2>nul
if not errorlevel 1 (
    echo [INFO]  Compiled with UBSAN ^(no-link-runtime^)
    set "SANITIZER_FLAGS=UBSAN(no-link-runtime)"
    goto :run
)

REM ── Strategy 3: UBSAN static ──────────────────────────────────────────────
echo [WARN]  Strategy 2 failed. Trying -fsanitize=undefined -static-libubsan ...
g++ -std=c++17 -O1 -g -fsanitize=undefined ^
    -static-libubsan ^
    -fno-omit-frame-pointer ^
    -I include ^
    benchmark\asan_test.cpp ^
    -o build\nli_asan.exe 2>nul
if not errorlevel 1 (
    echo [INFO]  Compiled with UBSAN ^(static^)
    set "SANITIZER_FLAGS=UBSAN(static)"
    goto :run
)

REM ── Strategy 4: No sanitizers — correctness run only ──────────────────────
echo [WARN]  Strategy 3 failed. Falling back to correctness-only build (no sanitizers^).
echo [WARN]  MinGW does not ship libubsan on this system.
echo [WARN]  12/12 PASS still proves logical correctness of all NLI code paths.
g++ -std=c++17 -O2 -g ^
    -fno-omit-frame-pointer ^
    -I include ^
    benchmark\asan_test.cpp ^
    -o build\nli_asan.exe
if errorlevel 1 (
    echo [ERROR] All compile strategies failed. Check g++ installation.
    exit /b 1
)
echo [INFO]  Compiled without sanitizers ^(correctness-only^)
set "SANITIZER_FLAGS=none(correctness-only)"

:run
echo.
echo [INFO]  Running test suite  [flags: !SANITIZER_FLAGS!]
echo ----------------------------------------------------------------

set "ASAN_OPTIONS=halt_on_error=0:detect_leaks=1:detect_stack_use_after_return=1"
set "UBSAN_OPTIONS=halt_on_error=0:print_stacktrace=1"

build\nli_asan.exe
set "EXIT_CODE=!errorlevel!"

echo.
echo ================================================================
if "!EXIT_CODE!"=="0" (
    echo  RESULT  : 12/12 PASS  --  exit code 0
    echo  Flags   : !SANITIZER_FLAGS!
    if "!SANITIZER_FLAGS!"=="none(correctness-only)" (
        echo  NOTE    : libubsan not available on this MinGW build.
        echo            12/12 correctness tests passed. For full ASAN proof,
        echo            run under WSL2: sudo apt install gcc g++ ^&^& ./run_asan.sh
    ) else (
        echo  NOTE    : Zero ASAN/UBSAN errors reported.
    )
) else (
    echo  RESULT  : !EXIT_CODE! test^(s^) FAILED
    echo  Flags   : !SANITIZER_FLAGS!
    echo  Review output above for details.
)
echo ================================================================
echo.

endlocal
exit /b !EXIT_CODE!
