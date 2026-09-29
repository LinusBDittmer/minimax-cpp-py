# Changelog

All notable changes to this project are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning
follows [SemVer](https://semver.org/); before 1.0.0, minor version bumps may
include breaking changes.

## [Unreleased]

## [0.1.2] - 2026-09-29

### Changed

- `verbose` now defaults to `1` (was `3`) across all three public entry
  points (`laplace_minimax`, `laplace_lp`, `biased_laplace`): only the run
  banner and errors are printed to stderr by default.

### Fixed

- `laplace_minimax` returned sub-optimal points and weights with a wrong
  `errmax` (≈ `ymin/ymax`) once `ymax/ymin` exceeded the saturation ratio of
  the given `nlap` (≈ 10 for `nlap=1`, 1e3 for `nlap=5`, 1e9 for `nlap=30`).
  Past that ratio the `[1, R]` minimax equals the `[1, ∞)` one and `x = R` is
  no longer an alternation point, but the Remez loop forced it to be one and
  converged to a spurious equioscillation that missed the last error
  extremum (e.g. `nlap=5`, `R=1e5`: true error 1.09e-2 instead of the
  optimal 6.43e-4). The Remez loop now detects this and switches to the
  saturated alternation set; the ext tables (and the regression reference)
  were regenerated. `biased_laplace` and `laplace_lp`, which warm-start from
  `laplace_minimax`, benefit as well.
- `errmax` returned by `laplace_minimax` and `biased_laplace` is now the
  maximum error on `[ymin, ymax]`, as documented. It was previously the
  error on the normalised interval `[1, ymax/ymin]`, i.e. off by a factor
  `ymin`.

## [0.1.1] - 2026-09-04

### Changed

- `laplace_minimax_ln` has been renamed to `laplace_lp`, and its `n` argument
  to `norm_p`, in both the C++ and Python API.
- `biased_laplace_minimax` has been renamed to `biased_laplace` in both the
  C++ and Python API.
- `biased_ud_laplace_minimax` (density-uncorrelated bias correction) has been
  merged into `biased_laplace` (formerly `biased_laplace_minimax`), which now
  applies the same correction unconditionally. `biased_ud_laplace_minimax` no
  longer exists as a separate function, in either the C++ or Python API.
- `verbose` now defaults to `3` (was `0`) across all three public entry
  points (`laplace_minimax`, `laplace_lp`, `biased_laplace`), so a run banner
  and per-iteration convergence tables are printed to stderr by default.
- The verbose-mode citation block printed by all three entry points now
  prints the actual reference (Helmich-Paris & Visscher, J. Comput. Phys.
  321, 2016) instead of a placeholder.

### Fixed

- A Debug-mode pivot-validity check inside `ddLuFactorize` could terminate
  the process instead of raising; it now behaves consistently with Release
  builds.
- Windows/macOS CI stability fixes (portable `M_PI`/`M_E` replacements for
  MSVC; `pip` self-upgrade invoked via `python -m pip`).
- `laplace_minimax` and `biased_laplace_minimax` now explicitly validate
  `nlap` (must be in `[1, 30]`) and `ymin`/`ymax` (must satisfy
  `0 < ymin < ymax`) at the Python binding layer, raising `ValueError` with a
  clear message. Previously `biased_laplace_minimax` had no such checks, and
  `laplace_minimax` relied on incidental failures deep in the table-lookup
  path — neither reliably rejected a non-positive or inverted `ymin`/`ymax`
  in Release builds, where the C++ core's own parameter assertions are
  compiled out.

## [0.1.0] - 2026-07-18

Initial release.

### Added

- Core MINIMAX-optimal Laplace quadrature (`laplace_minimax`) for `1/x` on
  `[ymin, ymax]`, `1 <= nlap <= 30`, via warm-started Remez/Maehly/Newton-
  Raphson iteration over pre-tabulated initial guesses.
- `laplace_minimax_ln`: L_n-norm-optimal quadrature (n >= 1), warm-started
  from the unbiased minimax solution.
- Density-weighted biasing: `DenominatorDensity` (FFT + quintic Hermite
  spline KDE over orbital energy denominators) and `biased_laplace_minimax`,
  including density-uncorrelated bias correction.
- Python bindings (`minimax_cpppy`, pybind11) and standalone C++ public API
  (`include/minimax_cpppy/`).
- Two-tier non-uniform lookup tables for `nlap = 1..30` (`gen_table`
  generator + `data_ext_NN` tables), ~6.3 MB total.
- CLI diagnostic tools (`minimax_loss_func`, `denominator_density`).
- C++ (CTest) and Python (pytest) test suites; C++ and Python usage examples.
