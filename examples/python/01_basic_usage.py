"""
01_basic_usage.py — Introduction to minimax_cpppy
==================================================

minimax_cpppy provides MINIMAX-optimal Laplace quadrature.  Given an interval
[ymin, ymax] and a number of quadrature points nlap, the library returns
exponents {a_k} and weights {w_k} such that the sum

    f(x) = sum_{k=0}^{nlap-1}  w_k * exp(-a_k * x)

approximates  1/x  on [ymin, ymax] with the smallest possible maximum
absolute error (minimax / Chebyshev equioscillation criterion).

Primary use in quantum chemistry: approximate orbital-energy denominators
so that expensive four-index sums factorise and reduce from O(N^5) to O(N^3).
See 02_lt_mp2_application.py for a full worked example.

Requirements
------------
    pip install minimax-cpppy numpy
"""

import numpy as np
import minimax_cpppy as mm   # the only external import needed


# =============================================================================
# 1. A single call
# =============================================================================

# Approximate  1/x  on [1.0, 1000.0] with nlap = 7 quadrature points.
#
# Rules of thumb:
#   • Larger ratio ymax/ymin  →  more points needed for the same accuracy.
#   • nlap ∈ {5, 7, 10} covers most molecular quantum-chemistry applications.
#   • Maximum supported nlap is 30.

nlap = 7
ymin = 1.0
ymax = 1000.0

exponents, weights, errmax = mm.laplace_minimax(nlap, ymin, ymax)
# ─────────────────────────────────────────────────────────────────────────────
#  exponents : numpy array of shape (nlap,), dtype float64
#              The a_k values in  f(x) = sum_k w_k * exp(-a_k * x)
#  weights   : numpy array of shape (nlap,), dtype float64
#              The w_k values
#  errmax    : float
#              Guaranteed maximum |1/x − f(x)| for any x in [ymin, ymax]
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print(f"laplace_minimax(nlap={nlap}, ymin={ymin}, ymax={ymax})")
print("=" * 60)
print(f"  Guaranteed max error (errmax) = {errmax:.3e}")
print()
print(f"  {'k':>3}   {'exponent a_k':>20}   {'weight w_k':>20}")
print(f"  {'─'*3}   {'─'*20}   {'─'*20}")
for k, (a, w) in enumerate(zip(exponents, weights)):
    print(f"  {k:>3}   {a:>20.14f}   {w:>20.14f}")
print()


# =============================================================================
# 2. Verify the approximation manually
# =============================================================================

# Build a dense logarithmic test grid on [ymin, ymax].
# Logarithmic spacing is important: 1/x varies most rapidly near ymin, so we
# want proportionally more test points there rather than spacing them linearly.
n_test = 500
t_grid = np.linspace(0.0, 1.0, n_test)          # uniform parameter in [0, 1]
x_grid = ymin * (ymax / ymin) ** t_grid          # x = ymin * (ymax/ymin)^t

# Evaluate f(x) = sum_k w_k * exp(-a_k * x) at all grid points at once.
#
# Broadcasting explanation:
#   x_grid    has shape (n_test,)    →  reshaped to  (n_test, 1)
#   exponents has shape (nlap,)      →  broadcast as  (1, nlap)
# The product  exponents * x_grid[:,None]  has shape (n_test, nlap).
# Summing over axis=1 (the k-axis) gives shape (n_test,).
approx = np.sum(
    weights[np.newaxis, :] * np.exp(-exponents[np.newaxis, :] * x_grid[:, np.newaxis]),
    axis=1
)

# True values and pointwise absolute errors
true_vals  = 1.0 / x_grid
abs_errors = np.abs(true_vals - approx)

print("=" * 60)
print(f"Verification on {n_test} log-spaced points in [{ymin}, {ymax}]")
print("=" * 60)
print(f"  Max absolute error (manual)  = {abs_errors.max():.3e}")
print(f"  Guaranteed max error (errmax) = {errmax:.3e}")
print(f"  Mean absolute error           = {abs_errors.mean():.3e}")
print()
# The manually computed maximum should be at most ~equal to errmax.
# Small differences are due to the finite sampling density of the grid.


# =============================================================================
# 3. Spot check at a specific x value
# =============================================================================

x_check   = 42.0
f_approx  = float(np.sum(weights * np.exp(-exponents * x_check)))
f_exact   = 1.0 / x_check
abs_err   = abs(f_exact - f_approx)

print("=" * 60)
print(f"Spot check at x = {x_check}")
print("=" * 60)
print(f"  Exact   1/x   = {f_exact:.14f}")
print(f"  Approx  f(x)  = {f_approx:.14f}")
print(f"  |error|       = {abs_err:.3e}   (errmax = {errmax:.3e})")
print()


# =============================================================================
# 4. Convergence: how many quadrature points do we need?
# =============================================================================

# More nlap → smaller error, but more exponential evaluations at runtime.
# The optimal nlap depends on the required accuracy and the width of the interval.

print("=" * 60)
print(f"Convergence: max error on [{ymin}, {ymax}] vs nlap")
print(f"(interval ratio ymax/ymin = {ymax/ymin:.0f})")
print("=" * 60)
print(f"  {'nlap':>5}   {'errmax (library)':>20}   {'error (manual)':>20}")
print(f"  {'─'*5}   {'─'*20}   {'─'*20}")

for n in [2, 3, 5, 7, 10, 15, 20]:
    try:
        e, w, err_lib = mm.laplace_minimax(n, ymin, ymax)

        # Evaluate on the same 500-point grid as above
        approx_n = np.sum(
            w[np.newaxis, :] * np.exp(-e[np.newaxis, :] * x_grid[:, np.newaxis]),
            axis=1
        )
        err_manual = np.abs(1.0 / x_grid - approx_n).max()

        print(f"  {n:>5}   {err_lib:>20.6e}   {err_manual:>20.6e}")

    except RuntimeError:
        # The library needs pre-tabulated initialisation data.  Very large nlap
        # values may not have data for every interval.
        print(f"  {n:>5}   (no tabulated initialisation data for this interval)")
print()


# =============================================================================
# 5. L_p-norm-optimal quadrature (laplace_lp)
# =============================================================================

# laplace_minimax minimises the L_inf (worst-case) error via equioscillation.
# laplace_lp instead minimises the L_p norm of the error for a real
# exponent norm_p >= 1:
#
#     ||eta||_p = (1/ymin) * ( integral |eta(t)|^p dt )^(1/p)
#
# Small norm_p spreads the error out (lower mean, higher peak); as norm_p
# grows the solution approaches the pure minimax (L_inf) result.  NOTE: the
# third return value here is the L_p NORM, not the L_inf error — so we also
# measure the true peak error on the grid for comparison.

print("=" * 60)
print(f"L_p-norm-optimal quadrature (nlap={nlap}, [{ymin}, {ymax}])")
print("=" * 60)
print(f"  {'norm_p':>6}   {'L_p norm':>18}   {'peak L_inf error':>18}")
print(f"  {'─'*6}   {'─'*18}   {'─'*18}")

for norm_p in [2.0, 4.0, 8.0, 16.0]:
    try:
        e_n, w_n, lp_norm = mm.laplace_lp(nlap, ymin, ymax, norm_p)
        approx_n = np.sum(
            w_n[np.newaxis, :] * np.exp(-e_n[np.newaxis, :] * x_grid[:, np.newaxis]),
            axis=1
        )
        peak = np.abs(1.0 / x_grid - approx_n).max()
        print(f"  {norm_p:>6.1f}   {lp_norm:>18.6e}   {peak:>18.6e}")
    except RuntimeError as exc:
        # Low non-integer norm_p (< ~1.25) can exhaust the continuation and fail.
        print(f"  {norm_p:>6.1f}   (RuntimeError: {exc})")

print(f"  (compare: pure minimax L_inf errmax = {errmax:.6e})")
print()


# =============================================================================
# 6. Error handling
# =============================================================================

print("=" * 60)
print("Error handling")
print("=" * 60)

# nlap = 0 is outside the supported range [1, 30]
try:
    mm.laplace_minimax(0, 1.0, 10.0)
except (ValueError, RuntimeError) as exc:
    print(f"  nlap=0     → {type(exc).__name__}: {exc}")

# ymax must be strictly greater than ymin
try:
    mm.laplace_minimax(5, 100.0, 1.0)
except (ValueError, RuntimeError) as exc:
    print(f"  ymax<ymin  → {type(exc).__name__}: {exc}")

# nlap > 30 is outside the supported range
try:
    mm.laplace_minimax(31, 1.0, 1000.0)
except ValueError as exc:
    print(f"  nlap=31    → ValueError: {exc}")

print()
print("Done.  See 02_lt_mp2_application.py for a quantum-chemistry application.")
