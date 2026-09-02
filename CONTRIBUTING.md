# Contributing to minimax-cpppy

## Building and testing

### C++

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMINIMAX_CPPPY_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Use `-DCMAKE_BUILD_TYPE=Debug` to enable the parameter-validation assertions
(`MINIMAX_CPPPY_DEBUG_MODE__`) — these are compiled out in Release. Run this
at least once locally before opening a PR that touches the C++ core; CI
also builds and tests Debug separately.

### Python

```bash
pip install -e . numpy pytest
pytest tests/python -v
```

## Before opening a PR

- Add or update tests for any behavior change (C++: `tests/*.cpp` + CTest
  registration in `tests/CMakeLists.txt`; Python: `tests/python/test_*.py`).
- Run both test suites above; CI runs Release + Debug C++ builds, an
  ASan/UBSan build, and Python tests across a platform/version matrix — a
  change that only passes Release locally may still fail CI.
- Keep the public C++ API (`include/minimax_cpppy/`) and Python bindings
  (`python/_minimax_core.cpp`) in sync: any new bound function should
  validate its own `nlap`/`ymin`/`ymax` (or equivalent) at the binding layer,
  since the C++ core intentionally skips validation in Release builds (see
  "Key constraints" in `CLAUDE.md`).
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes (new
  functions, behavior changes, bug fixes affecting output).

## Code style

- C++17, no exceptions across the Python/C++ boundary except
  `ValueError`/`RuntimeError` (via `py::value_error`/`std::runtime_error`).
- Internal numerics use the `DD` (double-double) type in `src/core/dd128.hpp`
  to avoid cancellation — don't drop to plain `double` inside the Remez/
  Maehly/Newton solvers without a specific reason.
- No new runtime dependencies without discussion — the C++ core has none by
  design; the Python layer depends only on NumPy.

## Regenerating lookup tables

Only needed if you change the Remez solver's convergence behavior or the
`[ymin, ymax]` grid. See "Regenerating ext tables" in `CLAUDE.md` — this
takes 30-60 minutes and rewrites `src/core/data_ext_*.{cpp,hpp}`.

## Architecture

See `CLAUDE.md` for the module layout and algorithm flow, and `README.md`
for usage documentation.

## License

By contributing, you agree your contributions are licensed under this
project's MIT license (see `LICENSE`).
