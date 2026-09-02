# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`minimax-cpppy` computes MINIMAX-optimal Laplace quadrature points for `1/x` on `[ymin, ymax]` — used in SOS-DF-LT-MP2 and related quantum chemistry methods. C++17 core library with pybind11 Python bindings.

## Build commands

### C++ library + tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMINIMAX_CPPPY_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Use `Debug` build type to activate parameter-validation assertions (`MINIMAX_CPPPY_DEBUG_MODE__`); Release compiles them out.

Run a single C++ test binary directly:
```bash
./build/tests/test_minimax_small
./build/tests/test_minimax_molecular
```

### Python bindings

Install for development (builds the pybind11 extension in-place via scikit-build-core):
```bash
pip install -e . numpy pytest
```

Run Python tests:
```bash
pytest tests/python -v
```

Run a single Python test:
```bash
pytest tests/python/test_bindings.py::test_smoke -v
```

### CLI diagnostic tools

```bash
cmake -B build_tools -DCMAKE_BUILD_TYPE=Debug -DMINIMAX_CPPPY_BUILD_TOOLS=ON -DMINIMAX_CPPPY_BUILD_TESTS=OFF
cmake --build build_tools --parallel
# Outputs: build_tools/minimax_loss_func, build_tools/denominator_density
```

### Python with CMake (standalone)

```bash
cmake -B build -DMINIMAX_CPPPY_BUILD_PYTHON=ON -DMINIMAX_CPPPY_BUILD_TESTS=OFF
cmake --build build --parallel
```

## Architecture

```
include/minimax_cpppy/minimax.hpp   ← public C++ API (MinimaxResult, laplaceMinimax)
include/minimax_cpppy/biasing.hpp   ← biased L∞ minimax via DenominatorDensity weighting
src/laplace.cpp                     ← thin shim: delegates to detail::laplaceMinimax
src/biasing.cpp                     ← biased Remez implementation
src/core/algorithm.hpp              ← init dispatch + full Remez (header-only, detail namespace)
src/core/remez.hpp                  ← remezLoop, logspaceInitFallback, computeInitialError
src/core/log_pretty.hpp             ← banner/table/citation formatting for verbose logging
src/core/dd128.hpp                  ← software 128-bit double-double arithmetic (DD type)
src/core/data_ext.hpp               ← interpolatedLookup, interpolateEntries, extTableForNlap, MAX_NLAP
src/core/data_ext_counts.{cpp,hpp}  ← EXT_COUNTS[30]: per-nlap variable row counts
src/core/data_ext_NN.{cpp,hpp}      ← two-tier non-uniform tables, nlap=01..30 (gen by gen_table)
src/core/grid.hpp                   ← cuspLog10 + two-tier buildLog10Grid (ratio grid for gen_table)
python/_minimax_core.cpp            ← pybind11 module (_minimax_core)
python/minimax_cpppy/__init__.py    ← re-exports laplace_minimax from _minimax_core
tools/gen_table.cpp                 ← table generator: runs Remez over the two-tier grid for nlap=1..30
tools/minimax_loss_func.cpp         ← diagnostic CLI: prints e(x) CSV on log grid
tools/denominator_density.cpp       ← diagnostic CLI: evaluates DenominatorDensity::evalW CSV
```

### Algorithm flow (`src/core/algorithm.hpp`)

`laplaceMinimax(nlap, ymin, ymax)`:
1. Call `data::interpolatedLookup(nlap, R)` (9-point Lagrange, log10-ratio space) where `R = ymax/ymin`.
2. Dispatch based on sampled approximation error `sampledErr`:
   - `sampledErr < 1e-12`: table holds a machine-precision solution; return directly.
   - `sampledErr ≤ 0.5` + valid extrema (bounds & sqrt-span check): warm-start Remez.
   - `sampledErr ≤ 0.5` but bad extrema: try `logspaceInitFallback(0.01)` → return LS if over-parameterised; else cold Maehly with table init.
   - `sampledErr > 0.5` (stale copy): same LS dispatch, falls back to `logspaceInitFallback(10.0)` seed.
3. Remez outer loop (up to `maxIter=200`):
   - **`maehly`** — Newton + Maehly deflation to find `2*nlap-1` extrema of `e'(x)`.
   - **`paraopt`** — Newton-Raphson to update `(expon, weight, error)` to satisfy equioscillation conditions; uses Armijo backtracking via `lnsrch`.
   - Stops when `paraopt` converges in a single NR step.
4. Scale results back from normalised interval `[1, R]` to `[ymin, ymax]`.

All internal arithmetic uses `DD` (double-double, ~32 significant decimal digits) to avoid cancellation.

### Regenerating ext tables

```bash
cmake -B build_tools -DCMAKE_BUILD_TYPE=Release -DMINIMAX_CPPPY_BUILD_TOOLS=ON -DMINIMAX_CPPPY_BUILD_TESTS=OFF
cmake --build build_tools --parallel
./build_tools/gen_table src/core          # writes data_ext_01..30.{cpp,hpp}, ~30-60 min
```

### Binary size note

Each `ExtTableEntry` is `213 doubles × 8 bytes = 1704 bytes`. The tables are
two-tier non-uniform (cusp-clustered) with variable per-nlap row counts stored
in `src/core/data_ext_counts.cpp` (`EXT_COUNTS[30]`, ~106–162 rows/nlap, 3873
rows total): `3873 × 1704 ≈ 6.3 MB` of initialized static data (vs the old
uniform 3000×30 ≈ 153 MB — a ~23× reduction). Lookup is binary search +
order-9 local Lagrange in `log10(range)` (`interpolateEntries`).

### Key constraints

- `1 ≤ nlap ≤ 30` — range covered by the ext tables.
- `0 < ymin < ymax` — checked only in Debug builds via `MINIMAX_CPPPY_DEBUG_MODE__`.
- Python bindings always raise `ValueError`/`RuntimeError`; C++ in Release builds does not validate.
