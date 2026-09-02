"""
05_biased_accuracy_comparison.py — Biased vs unbiased minimax on SOS-DF-LT-MP2
===============================================================================

Compares how quickly the biased and unbiased minimax Laplace quadratures
converge to the exact DF-SOS-MP2 correlation energy as a function of nlap.

Reference energy
----------------
The "exact" reference is the DF-SOS-MP2 energy obtained by evaluating the
quadrature error at zero — i.e., computing the double sum directly:

    E_SOS = -c_os * sum_{ia,jb}  (ia|jb)^2 / (D_ia + D_jb)

where  (ia|jb) = sum_P B_{ia}^P * B_{jb}^P  (density fitting).

This is exact within the DF approximation but costs O(N_ov^2) memory and
O(N_ov^2 * N_aux) flops — fine for a small molecule like H2O.

Compared quantities
-------------------
For nlap in {3, 5, 7, 10, 15}:

    unbiased: laplace_minimax(nlap, ymin, ymax)
    biased:   biased_laplace(nlap, ymin, ymax, eps_occ, eps_vir, bandwidth=1.0)

The biased scheme minimises the density-weighted error.  The KDE
bandwidth controls how peaked the weight function is: small bandwidth
(< 0.3) concentrates all equioscillation points at moderate Δ, relaxing
accuracy at ymin (HOMO-LUMO gap) and typically degrading SOS-MP2
accuracy relative to unbiased.  bandwidth ≈ 1.0 gives σ ≈ 25% of the
pair-denominator range, close enough to uniform that the biased Remez
produces a gentle, beneficial redistribution of equioscillation points.

Requirements
------------
    pip install pyscf minimax-cpppy numpy
"""

import numpy as np
from pyscf import gto, scf
from pyscf.df import addons as df_addons, incore as df_incore
import minimax_cpppy as mm


# =============================================================================
# 1. Build molecule and run RHF
# =============================================================================

mol = gto.Mole()
mol.atom = """
    O   0.00000   0.00000   0.00000
    H   0.75716   0.00000   0.58626
    H  -0.75716   0.00000   0.58626
"""
mol.basis   = 'cc-pVTZ'
mol.verbose = 3
mol.build()

mf = scf.RHF(mol)
mf.kernel()


# =============================================================================
# 2. Orbital energies and MO coefficients
# =============================================================================

mo_energy = mf.mo_energy
mo_coeff  = mf.mo_coeff
mo_occ    = mf.mo_occ

nao  = mol.nao_nr()
nmo  = mo_energy.size
nocc = mol.nelectron // 2
nvir = nmo - nocc

print(f"\nSystem:  H2O / cc-pVTZ")
print(f"  nao={nao},  nocc={nocc},  nvir={nvir},  n_ov={nocc*nvir}")

eps_occ = mo_energy[:nocc]
eps_vir = mo_energy[nocc:]
C_occ   = mo_coeff[:, :nocc]
C_vir   = mo_coeff[:, nocc:]

D_ia = eps_vir[np.newaxis, :] - eps_occ[:, np.newaxis]   # (nocc, nvir)
D_flat = D_ia.ravel()                                      # (nocc*nvir,)


# =============================================================================
# 3. DF 3-center integrals B_{ia}^P
# =============================================================================

auxbasis = 'cc-pVTZ-jkfit'
auxmol   = df_addons.make_auxmol(mol, auxbasis)
naux     = auxmol.nao_nr()

print(f"  aux: {auxbasis},  naux={naux}")

eri3c    = df_incore.aux_e2(mol, auxmol, 'int3c2e_sph', aosym='s1')   # (nao, nao, naux)
eri2c    = auxmol.intor('int2c2e_sph')                                  # (naux, naux)

evals, evecs = np.linalg.eigh(eri2c)
cutoff   = 1e-10 * evals.max()
keep     = evals > cutoff
J_inv_sqrt = (evecs[:, keep] * (1.0 / np.sqrt(evals[keep]))) @ evecs[:, keep].T

eri3c_mo = np.einsum('mnP,mi,na->iaP', eri3c, C_occ, C_vir)   # (nocc, nvir, naux)
B_iaP    = np.einsum('iaQ,QP->iaP', eri3c_mo, J_inv_sqrt)      # (nocc, nvir, naux)

# Flatten over (i, a): shape (nocc*nvir, naux)
B_flat = B_iaP.reshape(nocc * nvir, naux)


# =============================================================================
# 4. Reference: exact DF-SOS-MP2 by direct double sum
# =============================================================================
#
# (ia|jb) = sum_P B_{ia}^P B_{jb}^P  ->  ERI matrix shape (n_ov, n_ov)
# E_ref   = -c_os * sum_{m,n} ERI_{mn}^2 / (D_m + D_n)

c_os = 1.3

print("\nComputing exact DF-SOS-MP2 reference (direct sum)...")

ERI = B_flat @ B_flat.T                                   # (n_ov, n_ov)
denom = D_flat[:, np.newaxis] + D_flat[np.newaxis, :]    # (n_ov, n_ov)
E_ref = -c_os * np.sum(ERI ** 2 / denom)

print(f"  E_ref (DF-SOS-MP2 exact) = {E_ref:.8f} Eh")


# =============================================================================
# 5. DenominatorDensity for biased quadrature
# =============================================================================

density = mm.DenominatorDensity(eps_occ, eps_vir, bandwidth=1.0)
ymin, ymax = density.delta_min, density.delta_max

print(f"\nDenominator range: [{ymin:.4f}, {ymax:.4f}] Eh  (ratio={ymax/ymin:.1f})")


# =============================================================================
# 6. Laplace energy loop helper
# =============================================================================

def sos_lt_mp2_energy(expon, weight):
    """SOS-DF-LT-MP2 energy given Laplace nodes and weights."""
    n = len(expon)
    total = 0.0
    for k in range(n):
        scale  = np.exp(-0.5 * expon[k] * D_ia)          # (nocc, nvir)
        C_k    = B_iaP * scale[:, :, np.newaxis]          # (nocc, nvir, naux)
        Gamma  = np.einsum('iaP,iaQ->PQ', C_k, C_k)      # (naux, naux)
        total += weight[k] * np.einsum('PQ,PQ->', Gamma, Gamma)
    return -c_os * total


# =============================================================================
# 7. Convergence sweep
# =============================================================================

nlap_list = [3, 5, 7, 10, 15]

print("\n" + "=" * 80)
print("Convergence of DF-SOS-MP2 energy vs nlap")
print(f"  Reference: direct sum,  E_ref = {E_ref:.8f} Eh")
print("=" * 80)
print(f"  {'nlap':>5}   "
      f"{'E_unbiased (Eh)':>18}   {'ΔE_u (μEh)':>12}   "
      f"{'E_biased (Eh)':>18}   {'ΔE_b (μEh)':>12}   "
      f"{'improvement':>12}")
print(f"  {'─'*5}   {'─'*18}   {'─'*12}   {'─'*18}   {'─'*12}   {'─'*12}")

results = []

for n in nlap_list:
    try:
        e_u, w_u, err_u = mm.laplace_minimax(n, ymin, ymax)
        E_u = sos_lt_mp2_energy(e_u, w_u)
    except RuntimeError:
        E_u = None

    try:
        e_b, w_b, err_b = mm.biased_laplace(
            n, ymin, ymax, eps_occ, eps_vir, bandwidth=1.0)
        E_b = sos_lt_mp2_energy(e_b, w_b)
    except RuntimeError:
        E_b = None

    results.append((n, E_u, E_b))

    dE_u_uEh = (E_u - E_ref) * 1e6 if E_u is not None else None
    dE_b_uEh = (E_b - E_ref) * 1e6 if E_b is not None else None

    if E_u is not None and E_b is not None:
        factor = abs(dE_u_uEh) / abs(dE_b_uEh) if abs(dE_b_uEh) > 1e-12 else float('inf')
        impr_str = f"{factor:.2f}x" if factor < 1000 else ">1000x"
        print(f"  {n:>5}   "
              f"{E_u:>18.8f}   {dE_u_uEh:>+12.3f}   "
              f"{E_b:>18.8f}   {dE_b_uEh:>+12.3f}   "
              f"{impr_str:>12}")
    elif E_u is not None:
        print(f"  {n:>5}   "
              f"{E_u:>18.8f}   {dE_u_uEh:>+12.3f}   "
              f"{'(no data)':>18}   {'—':>12}   {'—':>12}")
    else:
        print(f"  {n:>5}   {'(no data)':>18}   {'—':>12}   "
              f"{'(no data)':>18}   {'—':>12}   {'—':>12}")

print()
print("  improvement = |ΔE_unbiased| / |ΔE_biased|   (>1 means biased is more accurate)")
print()

# =============================================================================
# 8. Per-quadrature-node energy contribution breakdown (nlap=7)
# =============================================================================
#
# Shows which Laplace nodes carry the largest weight and where biasing
# redistributes accuracy relative to the unbiased scheme.

n_show = 7
print("=" * 80)
print(f"Per-node energy contributions at nlap={n_show}")
print("=" * 80)

for label, get_quad in [("unbiased", lambda: mm.laplace_minimax(n_show, ymin, ymax)),
                         ("biased",   lambda: mm.biased_laplace(
                             n_show, ymin, ymax, eps_occ, eps_vir, bandwidth=1.0))]:
    try:
        expon, weight, _ = get_quad()
    except RuntimeError:
        print(f"  {label}: no tabulated data for nlap={n_show}")
        continue

    print(f"\n  {label}:")
    print(f"  {'k':>3}   {'t_k':>14}   {'w_k':>14}   {'w_k * ||Γ||²':>18}   {'cumul E_SOS (Eh)':>18}")
    print(f"  {'─'*3}   {'─'*14}   {'─'*14}   {'─'*18}   {'─'*18}")
    cumul = 0.0
    for k in range(n_show):
        scale = np.exp(-0.5 * expon[k] * D_ia)
        C_k   = B_iaP * scale[:, :, np.newaxis]
        Gamma = np.einsum('iaP,iaQ->PQ', C_k, C_k)
        contrib = weight[k] * np.einsum('PQ,PQ->', Gamma, Gamma)
        cumul  += contrib
        print(f"  {k:>3}   {expon[k]:>14.8f}   {weight[k]:>14.8f}   "
              f"{contrib:>18.10f}   {-c_os * cumul:>18.8f}")

print()
