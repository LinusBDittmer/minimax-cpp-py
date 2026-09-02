"""
04_biased_vs_unbiased.py — Biased vs standard minimax quadrature
=================================================================

Standard laplace_minimax minimises  max_{x in [ymin,ymax]} |1/x - f(x)|
uniformly — each point on [ymin, ymax] contributes equally to the error
criterion.

biased_laplace instead minimises the density-weighted error

    max_x  w(t(x)) * |1/x - f(x)|     where t(x) = ln(x/delta_min)

using the DenominatorDensity weight function w(t).  This relaxes the
error guarantee in sparsely populated regions (large denominators) and
tightens it where the density is high (near the HOMO-LUMO gap).

In a molecular MP2 energy calculation the dominant contribution to the
quadrature error comes from the high-density region, so the biased
result typically gives a more accurate correlation energy for the same
number of quadrature points.

Requirements
------------
    pip install minimax-cpppy numpy
"""

import numpy as np
import minimax_cpppy as mx

# H2O RHF/cc-pVDZ orbital energies in Hartree.
H2O_OCC  = np.array([-20.550438, -1.336658, -0.699262, -0.566562, -0.493142])
H2O_VIRT = np.array([0.185559, 0.256244, 0.789271, 0.854276, 1.163512,
                     1.200385, 1.253306, 1.444602, 1.476247, 1.674666,
                     1.867313, 1.934850, 2.452768, 2.490253, 3.285619,
                     3.338865, 3.510478, 3.865845, 4.147450])

NLAP = 6


# =============================================================================
# 1. Build the density and compute both sets of quadrature nodes
# =============================================================================

d = mx.DenominatorDensity(H2O_OCC, H2O_VIRT, bandwidth=1.0)
ymin, ymax = d.delta_min, d.delta_max

# Standard minimax: minimises uniform L-infinity error on [ymin, ymax]
expon_u, weight_u, errmax_u = mx.laplace_minimax(NLAP, ymin, ymax)

# Biased minimax: minimises density-weighted error; builds its own density
# internally from the same (occ, virt, bandwidth) inputs.
expon_b, weight_b, errmax_b = mx.biased_laplace(
    NLAP, ymin, ymax, H2O_OCC, H2O_VIRT, bandwidth=1.0)


# =============================================================================
# 2. Compare quadrature nodes
# =============================================================================

print("=" * 66)
print(f"Quadrature comparison: nlap={NLAP}, H2O, [ymin={ymin:.4f}, ymax={ymax:.4f}] Eh")
print("=" * 66)
print(f"  {'k':>3}   {'expon (unbiased)':>18}   {'expon (biased)':>18}   {'shift':>12}")
print(f"  {'─'*3}   {'─'*18}   {'─'*18}   {'─'*12}")
for k in range(NLAP):
    shift = expon_b[k] - expon_u[k]
    print(f"  {k:>3}   {expon_u[k]:>18.10f}   {expon_b[k]:>18.10f}   {shift:>+12.3e}")
print()
print(f"  {'k':>3}   {'weight (unbiased)':>18}   {'weight (biased)':>18}   {'shift':>12}")
print(f"  {'─'*3}   {'─'*18}   {'─'*18}   {'─'*12}")
for k in range(NLAP):
    shift = weight_b[k] - weight_u[k]
    print(f"  {k:>3}   {weight_u[k]:>18.10f}   {weight_b[k]:>18.10f}   {shift:>+12.3e}")
print()


# =============================================================================
# 3. Pointwise error comparison
# =============================================================================
#
# Evaluate |1/x - f(x)| at representative x values covering the full
# denominator range.  The interesting comparison is near the density peak
# (low t, near HOMO-LUMO gap) versus the sparse high-denominator tail.

def approx_err(expon, weight, x):
    f = np.sum(weight * np.exp(-expon * x))
    return abs(1.0 / x - f)

# t = ln(x/ymin); sample across [0, ln(ratio)]
t_max = np.log(d.ratio)
t_samples = np.array([0.02, 0.15, 0.30, 0.50, 0.65, 0.80, 0.98])
x_samples = ymin * np.exp(t_samples * t_max)

print("=" * 66)
print("Pointwise approximation error |1/x - f(x)|")
print("  t = ln(x/delta_min) / ln(ratio):  0 = small denom, 1 = large")
print("=" * 66)
print(f"  {'t (norm)':>9}   {'x (Eh)':>10}   {'err_unbiased':>14}   {'err_biased':>14}   {'ratio e_b/e_u':>14}")
print(f"  {'─'*9}   {'─'*10}   {'─'*14}   {'─'*14}   {'─'*14}")
for t_norm, x in zip(t_samples, x_samples):
    eu = approx_err(expon_u, weight_u, x)
    eb = approx_err(expon_b, weight_b, x)
    ratio_str = f"{eb/eu:.3f}" if eu > 0 else "—"
    better = "<" if eb < eu else ">"
    print(f"  {t_norm:>9.2f}   {x:>10.4f}   {eu:>14.3e}   {eb:>14.3e}   {better} 1  ({ratio_str})")
print()


# =============================================================================
# 4. errmax comparison
# =============================================================================
#
# errmax from biased_laplace is the density-weighted maximum error, not
# the plain L-infinity error.  Compute the unweighted max error manually so
# the comparison is on the same scale.

n_check = 2000
t_grid  = np.linspace(0.0, 1.0, n_check)
x_grid  = ymin * (ymax / ymin) ** t_grid
approx_u = np.sum(weight_u[:, np.newaxis] * np.exp(-expon_u[:, np.newaxis] * x_grid), axis=0)
approx_b = np.sum(weight_b[:, np.newaxis] * np.exp(-expon_b[:, np.newaxis] * x_grid), axis=0)
err_u_grid = np.abs(1.0 / x_grid - approx_u)
err_b_grid = np.abs(1.0 / x_grid - approx_b)

print("=" * 66)
print("Global error summary (unweighted L-infinity)")
print("=" * 66)
print(f"  Unbiased:  errmax (library) = {errmax_u:.3e}")
print(f"             errmax (manual)  = {err_u_grid.max():.3e}")
print()
print(f"  Biased:    errmax (library) = {errmax_b:.3e}  [density-weighted — not comparable]")
print(f"             errmax (manual)  = {err_b_grid.max():.3e}  [unweighted, for comparison]")
print()
print("  The biased errmax (library) reflects the weighted criterion and")
print("  cannot be compared directly to the unbiased errmax.")
print()

# Find where each scheme has the larger error (crossing point)
where_biased_better = err_b_grid < err_u_grid
frac_biased_better = where_biased_better.mean()
print(f"  Biased has smaller error at {frac_biased_better*100:.0f}% of sampled x values.")
print(f"  The region where biased wins corresponds to the density peak,")
print(f"  which is system-dependent — not necessarily near small denominators.")
