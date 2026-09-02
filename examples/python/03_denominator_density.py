"""
03_denominator_density.py — The DenominatorDensity object
===========================================================

In MP2/CC theory the orbital-energy denominator for pair (ia, jb) is

    Delta_{ia,jb} = D_{ia} + D_{jb}    where   D_{ia} = eps_a - eps_i > 0

Standard minimax quadrature minimises the L-infinity error uniformly over
[delta_min, delta_max], treating all denominator values equally.  But the
denominators are NOT uniformly distributed: near the HOMO-LUMO gap (small
D_{ia}) there are many pairs; at large D_{ia} there are few.  A biased
quadrature that concentrates accuracy where the density is high reduces the
physical error in the correlation energy more efficiently.

DenominatorDensity computes this density via FFT-based KDE over the pairwise
values 2*(eps_a - eps_i), then fits a quintic Hermite spline for smooth
evaluation.  This example shows what that density captures.

Requirements
------------
    pip install minimax-cpppy numpy
"""

import numpy as np
import minimax_cpppy as mx

# H2O RHF/cc-pVDZ orbital energies in Hartree.
# Source: PySCF calculation — same values used in the test suite.
H2O_OCC  = np.array([-20.550438, -1.336658, -0.699262, -0.566562, -0.493142])
H2O_VIRT = np.array([0.185559, 0.256244, 0.789271, 0.854276, 1.163512,
                     1.200385, 1.253306, 1.444602, 1.476247, 1.674666,
                     1.867313, 1.934850, 2.452768, 2.490253, 3.285619,
                     3.338865, 3.510478, 3.865845, 4.147450])


# =============================================================================
# 1. Pairwise denominator distribution — computed directly from orbitals
# =============================================================================
#
# Each occupied orbital i and virtual orbital a contributes one single-particle
# gap D_{ia} = eps_a - eps_i.  The pairwise denominator for (ia, jb) is
# D_{ia} + D_{jb}.  Its range is [2*D_min, 2*D_max].
#
# We compute all D_{ia} explicitly to visualise the distribution before
# invoking DenominatorDensity.

# Broadcasting: D_ia[i, a] = eps_virt[a] - eps_occ[i]
D_ia = H2O_VIRT[np.newaxis, :] - H2O_OCC[:, np.newaxis]   # (nocc, nvir)
D_flat = D_ia.flatten()

# Pairwise denominators are Delta = D_{ia} + D_{jb}; the range is:
delta_min_phys = 2.0 * D_flat.min()   # both pairs at the HOMO-LUMO gap
delta_max_phys = 2.0 * D_flat.max()   # both pairs at the spectral extremes

print("=" * 62)
print("H2O orbital energy denominators D_{ia} = eps_a - eps_i")
print("=" * 62)
print(f"  n_occ  = {len(H2O_OCC)},  n_virt = {len(H2O_VIRT)}")
print(f"  Total D_{{ia}} values: {len(D_flat)}")
print(f"  D_min  = {D_flat.min():.6f} Eh   (HOMO-LUMO gap)")
print(f"  D_max  = {D_flat.max():.6f} Eh")
print(f"  Pairwise delta_min = 2 * D_min = {delta_min_phys:.6f} Eh")
print(f"  Pairwise delta_max = 2 * D_max = {delta_max_phys:.6f} Eh")
print(f"  Ratio  delta_max / delta_min   = {delta_max_phys / delta_min_phys:.2f}")
print()

# Text histogram in log-denominator space.
# t = ln(D_ia / D_min) maps D values to [0, ln(D_max/D_min)].
# Plotting in t-space makes sense because minimax_cpppy works in this
# log-scaled coordinate internally.
n_bins = 10
t_max_D = np.log(D_flat.max() / D_flat.min())
t_D = np.log(D_flat / D_flat.min())
counts, edges = np.histogram(t_D, bins=n_bins)
bar_max = 28

print("  Distribution of D_{ia} in log-denominator space")
print("  (t = ln(D_ia / D_min),  each bar = one histogram bin)")
print()
print(f"  {'t range':>15}    {'count':>5}   histogram")
print(f"  {'─'*15}    {'─'*5}   {'─'*bar_max}")
for i in range(n_bins):
    bar_len = int(counts[i] * bar_max / counts.max()) if counts.max() > 0 else 0
    label = f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
    print(f"  {label:>15}    {counts[i]:>5}   {'█' * bar_len}")
print()
print("  H2O: D_{ia} peaks at mid-range t (core 1s orbital creates a")
print("  second high-t cluster).  DenominatorDensity uses the KDE of")
print("  PAIRWISE gaps — the convolution of this distribution with itself.")
print("  Run 04_biased_vs_unbiased.py to see where the optimizer focuses.")
print()


# =============================================================================
# 2. Construct DenominatorDensity and verify it finds the same bounds
# =============================================================================
#
# DenominatorDensity takes the raw orbital energies and internally:
#   1. Computes delta_min = 2*(eps_virt_min - eps_occ_max)
#      and    delta_max = 2*(eps_virt_max - eps_occ_min)
#   2. Builds a KDE of the pairwise denominator distribution via FFT
#   3. Fits a quintic Hermite spline for smooth, differentiable evaluation
#
# bandwidth is the KDE Gaussian sigma expressed as a fraction of
# ln(delta_max/delta_min).  Larger bandwidth = smoother density.

d = mx.DenominatorDensity(H2O_OCC, H2O_VIRT, bandwidth=1.0)

print("=" * 62)
print("DenominatorDensity (bandwidth=1.0)")
print("=" * 62)
print(f"  d.delta_min = {d.delta_min:.6f} Eh")
print(f"  d.delta_max = {d.delta_max:.6f} Eh")
print(f"  d.ratio     = {d.ratio:.6f}")
print()
print(f"  Manual delta_min = {delta_min_phys:.6f} Eh  (match: {abs(d.delta_min - delta_min_phys) < 1e-6})")
print(f"  Manual delta_max = {delta_max_phys:.6f} Eh  (match: {abs(d.delta_max - delta_max_phys) < 1e-6})")
print()


# =============================================================================
# 3. Effect of bandwidth on the density shape
# =============================================================================
#
# The bandwidth parameter controls smoothing.  A very narrow bandwidth
# produces a density that closely tracks the discrete histogram above;
# a wide bandwidth produces a near-uniform density, making the biased
# optimisation approach the unbiased (standard minimax) result.

print("=" * 62)
print("Bandwidth sensitivity")
print("  (all bandwidths give the same bounds; the density SHAPE changes)")
print("=" * 62)
print(f"  {'bandwidth':>10}   {'delta_min (Eh)':>16}   {'delta_max (Eh)':>16}   {'ratio':>8}")
print(f"  {'─'*10}   {'─'*16}   {'─'*16}   {'─'*8}")
for bw in [0.5, 1.0, 2.0]:
    db = mx.DenominatorDensity(H2O_OCC, H2O_VIRT, bandwidth=bw)
    print(f"  {bw:>10.1f}   {db.delta_min:>16.6f}   {db.delta_max:>16.6f}   {db.ratio:>8.4f}")
print()
print("  Bounds are invariant to bandwidth — only the density shape changes.")
print("  Use the denominator_density CLI tool to visualise evalW(t) directly:")
print()
print("    build_tools/denominator_density <orbital_file> 100 1.0")
print()
print("See 04_biased_vs_unbiased.py for how this density shapes the quadrature.")
