#!/usr/bin/env bash
# =============================================================================
# run_all.sh — NLI v5.0 Full Benchmark Suite
# Works on: Windows (Git Bash / MSYS2), Linux, macOS
#
# Usage:
#   bash run_all.sh [sosd_data_dir] [results_dir] [size1 size2 ...]
#
# Examples:
#   bash run_all.sh sosd_data results               # all three sizes
#   bash run_all.sh sosd_data results 100000        # quick test only
#   bash run_all.sh sosd_data results 100000 1000000
#
# Defaults:
#   sosd_data_dir = sosd_data
#   results_dir   = results
#   sizes         = 100000 1000000 200000000
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${1:-sosd_data}"
OUT_DIR="${2:-results}"
BUILD_DIR="build"
FIG_DIR="figures"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { printf "\n${GREEN}══ %s ══${NC}\n" "$*"; }

# ── Detect OS — set binary extension ─────────────────────────────────────────
# On Windows (Git Bash / MSYS2) OSTYPE is "msys" or "cygwin".
# MinGW g++ automatically appends .exe even when -o omits it.
# We must use the .exe form explicitly so the right binary runs.
case "$OSTYPE" in
    msys*|cygwin*|win32*)
        EXE=".exe"
        OS_NAME="Windows (Git Bash / MSYS2)"
        ;;
    *)
        EXE=""
        OS_NAME="Linux / macOS"
        ;;
esac
info "Platform: $OS_NAME  (binary suffix: '${EXE:-none}')"

# ── Key sizes ─────────────────────────────────────────────────────────────────
# If sizes passed as args → use them directly (no menu).
# Otherwise → show interactive selection menu.
if [[ $# -gt 2 ]]; then
    MAIN_SIZES=("${@:3}")
else
    echo ""
    echo "  ┌─────────────────────────────────────────────────┐"
    echo "  │         Select key sizes to benchmark           │"
    echo "  │                                                 │"
    echo "  │  1)  100K only       (~1–2 min, low RAM)        │"
    echo "  │  2)  1M only         (~3–5 min, moderate RAM)   │"
    echo "  │  3)  100K + 1M       (~5–7 min, moderate RAM)   │"
    echo "  │  4)  100K + 1M + 200M  (FULL — needs 8+ GB RAM) │"
    echo "  └─────────────────────────────────────────────────┘"
    echo ""
    printf "  Enter choice [1-4]: "
    read -r CHOICE </dev/tty
    case "$CHOICE" in
        1) MAIN_SIZES=(100000) ;;
        2) MAIN_SIZES=(1000000) ;;
        3) MAIN_SIZES=(100000 1000000) ;;
        4) MAIN_SIZES=(100000 1000000 200000000) ;;
        *) error "Invalid choice '$CHOICE'. Run again and enter 1, 2, 3, or 4." ;;
    esac
fi

# Drift benchmark: skip 100K (too small for meaningful stable/drift phases);
# use 1M and above from whatever the user requested.
DRIFT_SIZES=()
for sz in "${MAIN_SIZES[@]}"; do
    [[ $sz -ge 1000000 ]] && DRIFT_SIZES+=("$sz")
done
[[ ${#DRIFT_SIZES[@]} -eq 0 ]] && DRIFT_SIZES=(1000000)

info "Key sizes (main + ablation): ${MAIN_SIZES[*]}"
info "Key sizes (drift):           ${DRIFT_SIZES[*]}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
command -v g++ >/dev/null 2>&1 || error "g++ not found. Install MinGW (MSYS2: pacman -S mingw-w64-x86_64-gcc) and add its bin/ to PATH."

# Resolve Python — Windows names it 'python', Linux/macOS 'python3'
PYTHON=""
for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        # Make sure it's actually Python 3, not Python 2
        ver=$("$py" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo 0)
        if [[ "$ver" == "3" ]]; then
            PYTHON="$py"
            break
        fi
    fi
done
[[ -n "$PYTHON" ]] || error "Python 3 not found. Install Python 3 and add to PATH."
info "Python: $PYTHON  ($($PYTHON --version 2>&1))"

for ds in books_200M_uint64 fb_200M_uint64 wiki_ts_200M_uint64; do
    if [[ -f "$DATA_DIR/$ds" ]]; then
        sz=$(wc -c < "$DATA_DIR/$ds" 2>/dev/null || echo "?")
        info "Dataset OK: $DATA_DIR/$ds  ($(( ${sz:-0} / 1048576 )) MB)"
    else
        warn "Dataset MISSING: $DATA_DIR/$ds  — that dataset will be skipped."
    fi
done

mkdir -p "$BUILD_DIR" "$OUT_DIR" "$FIG_DIR"

# Install Python dependencies silently (won't fail if already installed)
info "Checking Python dependencies..."
$PYTHON -m pip install --quiet --upgrade matplotlib pandas numpy 2>/dev/null \
    || warn "pip install failed — figures may not generate. Run: pip install matplotlib pandas numpy"

# ── Compile ───────────────────────────────────────────────────────────────────
step "Step 1/4 — Compile"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O3 -march=native -mpopcnt -funroll-loops -ffast-math -I include"

compile() {
    local src="$1" out="$2" label="$3"
    info "Compiling $label..."
    # Remove any stale Linux ELF binary that would shadow the Windows .exe
    rm -f "$BUILD_DIR/$out"
    # Compile — do NOT pipe (piping loses the exit code on Windows)
    if "$CXX" $CXXFLAGS "$src" -o "$BUILD_DIR/${out}${EXE}" 2>&1 | grep -v "optional but"; then
        :   # grep may exit 1 if no lines match, so we ignore its exit code
    fi
    # Confirm the binary was actually produced
    if [[ ! -f "$BUILD_DIR/${out}${EXE}" ]]; then
        error "Compile FAILED: $label — $BUILD_DIR/${out}${EXE} not created."
    fi
    info "$label OK → $BUILD_DIR/${out}${EXE}"
}

compile "benchmark/main_benchmark.cpp"     "nli_benchmark" "nli_benchmark"
compile "benchmark/drift_benchmark.cpp"    "nli_drift"     "nli_drift"
compile "benchmark/ablation_benchmark.cpp" "nli_ablation"  "nli_ablation"

BIN_BENCH="$BUILD_DIR/nli_benchmark${EXE}"
BIN_DRIFT="$BUILD_DIR/nli_drift${EXE}"
BIN_ABL="$BUILD_DIR/nli_ablation${EXE}"

# ── Step 2: Main benchmark ────────────────────────────────────────────────────
step "Step 2/4 — Main Benchmark"
info "Indexes: B-Tree / ALEX / PGM / RMI / NLI-Full / NLI-Linear / NLI-NoDrift"
info "Sizes:   ${MAIN_SIZES[*]}"
info "Query scaling: 50K (100K keys) | 100K (1M) | 500K (200M)  [inserts: 10K / 20K / 100K]"
echo ""
{ time "$BIN_BENCH" "$DATA_DIR" "$OUT_DIR" "${MAIN_SIZES[@]}"; } 2>&1
info "Saved → $OUT_DIR/benchmark_results.csv"

# ── Step 3: Drift benchmark ───────────────────────────────────────────────────
step "Step 3/4 — Drift Detection Benchmark"
info "Sizes:   ${DRIFT_SIZES[*]}"
info "5 drift scenarios × 3 datasets × ${#DRIFT_SIZES[@]} size(s) = $(( 5 * 3 * ${#DRIFT_SIZES[@]} )) experiments"
echo ""
{ time "$BIN_DRIFT" "$DATA_DIR" "$OUT_DIR" "${DRIFT_SIZES[@]}"; } 2>&1
info "Saved → $OUT_DIR/drift_results.csv"

# ── Step 4: Ablation ─────────────────────────────────────────────────────────
step "Step 4/4 — Ablation Study"
info "Sizes: ${MAIN_SIZES[*]}"
echo ""
{ time "$BIN_ABL" "$DATA_DIR" "$OUT_DIR" "${MAIN_SIZES[@]}"; } 2>&1
info "Saved → $OUT_DIR/ablation_results.csv"

# ── Step 5: Figures + report ──────────────────────────────────────────────────
step "Step 5 — Figures & Report"

$PYTHON generate_figures.py "$OUT_DIR" "$FIG_DIR"
if [[ $? -ne 0 ]]; then
    warn "generate_figures.py reported an error — check Python/matplotlib."
else
    info "Figures and report generated successfully."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
step "Complete!"
echo ""
echo "  Results:"
for f in "$OUT_DIR/benchmark_results.csv" "$OUT_DIR/drift_results.csv" \
          "$OUT_DIR/ablation_results.csv"  "$OUT_DIR/benchmark_report.md"; do
    if [[ -f "$f" ]]; then
        printf "    %-50s  OK\n" "$f"
    else
        printf "    %-50s  MISSING\n" "$f"
    fi
done
echo ""
echo "  Figures (${FIG_DIR}/):"
for f in "$FIG_DIR"/*.png; do
    [[ -f "$f" ]] && printf "    %s\n" "$(basename "$f")"
done
echo ""
info "Run complete. Open $OUT_DIR/benchmark_report.md for the full report."
