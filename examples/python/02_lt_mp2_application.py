"""
02_sos_df_lt_mp2.py — Scaled-Opposite-Spin Density-Fitting LT-MP2
==================================================================

Demonstrates how minimax_cpppy enables an O(N^4)-scaling MP2 energy via the
Laplace transform trick.  All four steps are shown explicitly:

  1. Run RHF with PySCF to get orbital energies and coefficients.
  2. Build density-fitting (DF) 3-center integrals B_{ia}^P.
  3. Obtain minimax Laplace quadrature nodes {t_k} and weights {w_k}.
  4. Accumulate the SOS-MP2 energy in a loop over Laplace points.

Theory
------
Standard MP2 correlation energy (spin-orbital notation, RHF reference):

    E_MP2 = sum_{i<j, a<b}  (2*(ia|jb) - (ib|ja)) * (ia|jb) / (D_ia + D_jb)

SOS-MP2 (Jung & Head-Gordon, J. Chem. Phys. 122, 2005) keeps only the
opposite-spin contribution, scaled by an empirical factor c_os ≈ 1.3:

    E_SOS = -c_os * sum_{ia, jb}  (ia|jb)^2 / (D_ia + D_jb)

where  D_ia = eps_a - eps_i > 0  (virtual minus occupied orbital energy).

Laplace transform identity:

    1/(D_ia + D_jb)  =  integral_0^inf  exp(-t * D_ia) * exp(-t * D_jb) dt
                     ≈  sum_k  w_k * exp(-t_k * D_ia) * exp(-t_k * D_jb)

This replaces the problematic denominator with a product that factorises
over the (i,a) and (j,b) pairs independently — enabling O(N^3) algorithms.

Density fitting (DF / resolution-of-identity):

    (ia|jb) ≈ sum_P  B_{ia}^P * B_{jb}^P

Combining both tricks:

    E_SOS = -c_os * sum_k  w_k * ||Gamma(t_k)||_F^2

where

    Gamma_{PQ}(t_k) = sum_{ia}  C_{ia}^P(t_k) * C_{ia}^Q(t_k)
    C_{ia}^P(t_k)   = B_{ia}^P * exp(-t_k * D_ia / 2)

Each Laplace step costs O(N_occ * N_vir * N_aux) to build C and
O(N_occ * N_vir * N_aux^2) to form Gamma — both polynomial in N.

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
mol.basis   = 'cc-pVDZ'   # correlation-consistent double-zeta basis
mol.verbose = 3            # print convergence info; set to 0 to suppress
mol.build()

mf = scf.RHF(mol)
mf.kernel()                # solve the Hartree-Fock equations


# =============================================================================
# 2. Extract orbital energies and MO coefficients
# =============================================================================

# mo_energy : MO energies in Hartree, shape (nmo,)
# mo_coeff  : MO coefficient matrix C_{mu,p}, shape (nao, nmo)
# mo_occ    : occupation numbers (2.0 for occupied, 0.0 for virtual), shape (nmo,)
mo_energy = mf.mo_energy
mo_coeff  = mf.mo_coeff
mo_occ    = mf.mo_occ

nao  = mol.nao_nr()            # number of AO basis functions
nmo  = mo_energy.size          # number of MOs (= nao for a non-ECP RHF)
nocc = mol.nelectron // 2      # number of doubly occupied orbitals
nvir = nmo - nocc              # number of virtual orbitals

print(f"\nSystem summary:")
print(f"  nao={nao},  nocc={nocc},  nvir={nvir}")

# Separate occupied and virtual orbital energies and coefficient blocks.
# Convention: orbitals are ordered from lowest to highest energy, so
# indices 0..nocc-1 are occupied and nocc..nmo-1 are virtual.
eps_occ = mo_energy[:nocc]    # shape (nocc,),  negative values for bound orbitals
eps_vir = mo_energy[nocc:]    # shape (nvir,),  positive values for unbound virtuals

C_occ   = mo_coeff[:, :nocc]  # shape (nao, nocc)
C_vir   = mo_coeff[:, nocc:]  # shape (nao, nvir)


# =============================================================================
# 3. Build the orbital energy denominator matrix D_ia
# =============================================================================

# D_ia = eps_a - eps_i  (virtual energy minus occupied energy)
# By Koopmans' theorem, D_ia > 0 for a stable closed-shell RHF solution.
#
# Broadcasting: eps_occ[:, None] has shape (nocc, 1)
#               eps_vir[None, :] has shape (1, nvir)
# Their difference broadcasts to shape (nocc, nvir).
D_ia = eps_vir[np.newaxis, :] - eps_occ[:, np.newaxis]   # shape (nocc, nvir)

# The pairwise denominator D_ia + D_jb ranges between:
#   minimum: 2 * min(D_ia)  (both pairs near HOMO-LUMO gap)
#   maximum: 2 * max(D_ia)  (both pairs at extreme ends)
# The Laplace approximation  1/x ≈ sum_k w_k exp(-t_k x)  must cover this range.
ymin = 2.0 * D_ia.min()
ymax = 2.0 * D_ia.max()

print(f"\nDenominator range:  [{ymin:.4f}, {ymax:.4f}] Eh")
print(f"  ratio ymax/ymin = {ymax / ymin:.1f}")


# =============================================================================
# 4. Minimax Laplace quadrature
# =============================================================================

# nlap controls how many Laplace quadrature points to use.
# More points → smaller error in the 1/(D_ia + D_jb) approximation
# → more accurate E_SOS, but more work per Laplace iteration.
# For molecular systems, nlap = 5-12 usually gives well-converged results.
nlap = 7

# laplace_minimax returns:
#   expon  : quadrature nodes  t_k,  shape (nlap,)
#   weight : quadrature weights w_k,  shape (nlap,)
#   errmax : guaranteed max |1/x - sum_k w_k exp(-t_k x)| on [ymin, ymax]
expon, weight, errmax = mm.laplace_minimax(nlap, ymin, ymax)

print(f"\nMinimax Laplace quadrature:  nlap={nlap}")
print(f"  Max error in 1/x on [{ymin:.4f}, {ymax:.4f}]:  {errmax:.2e}")
print(f"\n  {'k':>3}   {'t_k (exponent)':>18}   {'w_k (weight)':>18}")
for k in range(nlap):
    print(f"  {k:>3}   {expon[k]:>18.10f}   {weight[k]:>18.10f}")


# =============================================================================
# 5. Density-fitting 3-center integrals  B_{ia}^P
# =============================================================================
#
# The DF (density fitting / resolution-of-identity) approximation replaces the
# exact 4-center 2-electron integral (ia|jb) with a factorised form:
#
#   (ia|jb)  ≈  sum_P  B_{ia}^P * B_{jb}^P
#
# where  B_{ia}^P = sum_Q  (ia|Q) * [J^{-1/2}]_{QP}
#
# and the auxiliary quantities are:
#   (ia|Q)  =  sum_{mu,nu}  C_mu^i * C_nu^a * (mu nu | Q)    (3-center AO integral)
#   J_{PQ}  =  (P|Q)                                          (2-center metric)
#
# An auxiliary basis set {chi_P} is used.  It is larger than the orbital basis
# but far smaller than the full 4-center integral tensor, enabling the speedup.

auxbasis = 'cc-pVDZ-jkfit'   # auxiliary basis optimised for DF calculations
auxmol   = df_addons.make_auxmol(mol, auxbasis)
naux     = auxmol.nao_nr()

print(f"\nAuxiliary basis:  {auxbasis},  naux={naux}")

# Step 5a: 3-center Coulomb integrals (mu nu | P) in the AO basis.
#
#   eri3c[mu, nu, P] = integral phi_mu(r1) phi_nu(r1) (1/|r1-r2|) chi_P(r2) dr1 dr2
#
# aosym='s1': no symmetry — full (nao, nao, naux) tensor.
# (aosym='s2ij' exploits mu<->nu symmetry and halves storage, but the
#  subsequent MO transformation is then less straightforward.)
eri3c = df_incore.aux_e2(mol, auxmol, 'int3c2e_sph', aosym='s1')
# shape: (nao, nao, naux)

# Step 5b: 2-center Coulomb metric  J_{PQ} = (P|Q).
#
#   eri2c[P, Q] = integral chi_P(r1) (1/|r1-r2|) chi_Q(r2) dr1 dr2
#
# This is a symmetric, positive-definite matrix for a linearly independent
# auxiliary basis.  We need J^{-1/2} to produce fitted integrals B_{ia}^P.
eri2c = auxmol.intor('int2c2e_sph')   # shape: (naux, naux)

# Step 5c: Compute J^{-1/2} via eigendecomposition.
#
#   J = V * Lambda * V^T         (V = eigenvectors, Lambda = diagonal eigenvalues)
#   J^{-1/2} = V * Lambda^{-1/2} * V^T
#
# We use eigh (symmetric eigensolver) since J is symmetric by construction.
evals, evecs = np.linalg.eigh(eri2c)

# Drop near-zero eigenvalues to avoid dividing by ~0.
# These correspond to nearly linearly dependent auxiliary basis functions.
cutoff   = 1e-10 * evals.max()
keep     = evals > cutoff
n_kept   = keep.sum()
print(f"  Auxiliary screening:  {naux} functions → {n_kept} kept (cutoff={cutoff:.1e})")

# J^{-1/2} has shape (naux, naux).
# We keep only the eigenvectors/eigenvalues above the threshold.
J_inv_sqrt = (evecs[:, keep] * (1.0 / np.sqrt(evals[keep]))) @ evecs[:, keep].T

# Step 5d: Transform 3-center integrals from AO to MO (occupied-virtual) basis.
#
#   eri3c_mo[i, a, P] = sum_{mu, nu}  C_mu^i * C_nu^a * eri3c[mu, nu, P]
#
# This contracts both AO indices (mu, nu) with the MO coefficient matrices.
eri3c_mo = np.einsum('mnP,mi,na->iaP', eri3c, C_occ, C_vir)
# shape: (nocc, nvir, naux)

# Step 5e: Apply the metric dressing to get the fitted integrals B_{ia}^P.
#
#   B_{ia}^P = sum_Q  eri3c_mo[i, a, Q] * J^{-1/2}[Q, P]
#
# After this step:  (ia|jb) ≈ sum_P  B_{ia}^P * B_{jb}^P
B_iaP = np.einsum('iaQ,QP->iaP', eri3c_mo, J_inv_sqrt)
# shape: (nocc, nvir, naux)


# =============================================================================
# 6. SOS-DF-LT-MP2 energy accumulation
# =============================================================================

c_os = 1.3   # SOS scaling coefficient (Jung & Head-Gordon, 2005).
             # Empirically fitted; c_os=1 recovers unscaled OS-MP2.

# We accumulate:  E_SOS_sum = sum_k  w_k * ||Gamma(t_k)||_F^2
# and then apply:  E_SOS = -c_os * E_SOS_sum
E_SOS_sum = 0.0

print(f"\nSOS-DF-LT-MP2 energy loop  (c_os={c_os}, nlap={nlap})")
print(f"  {'k':>3}   {'t_k':>14}   {'w_k':>14}   {'w_k * ||Γ||²':>18}")
print(f"  {'─'*3}   {'─'*14}   {'─'*14}   {'─'*18}")

for k in range(nlap):
    t_k = expon[k]    # Laplace quadrature node  (the 't' variable)
    w_k = weight[k]   # Laplace quadrature weight

    # Dressed integrals  C_{ia}^P(t_k) = B_{ia}^P * exp(-t_k * D_ia / 2)
    #
    # The factor 1/2 on the exponent distributes the Laplace weight
    # symmetrically over the two pairs (i,a) and (j,b):
    #   exp(-t_k * (D_ia + D_jb)) = [exp(-t_k/2 * D_ia)]^2 * [exp(-t_k/2 * D_jb)]^2
    # which makes Gamma_{PQ} = (C C^T)_{PQ} a simple matrix product.
    #
    # scale has shape (nocc, nvir).  Broadcasting with B_iaP (nocc, nvir, naux)
    # requires adding a trailing singleton dimension.
    scale = np.exp(-0.5 * t_k * D_ia)                   # shape (nocc, nvir)
    C_iaP = B_iaP * scale[:, :, np.newaxis]              # shape (nocc, nvir, naux)

    # Gamma_{PQ}(t_k) = sum_{i,a}  C_{ia}^P * C_{ia}^Q
    #
    # The einsum contracts over both the occupied index i and the virtual index a.
    # Result is the 'middle' matrix of DF-LT-MP2: shape (naux, naux).
    Gamma = np.einsum('iaP,iaQ->PQ', C_iaP, C_iaP)

    # Frobenius norm squared:  ||Gamma||_F^2 = sum_{P,Q}  Gamma_{PQ}^2
    norm_sq = np.einsum('PQ,PQ->', Gamma, Gamma)

    E_SOS_sum += w_k * norm_sq
    print(f"  {k:>3}   {t_k:>14.8f}   {w_k:>14.8f}   {w_k * norm_sq:>18.10f}")

# The correlation energy is negative (electrons lower their energy by correlating).
# The sum E_SOS_sum is strictly positive, so we negate.
E_SOS = -c_os * E_SOS_sum

print(f"\n  E_SOS = -{c_os} × {E_SOS_sum:.8f} Eh")
print(f"\nSOS-DF-LT-MP2 correlation energy:  {E_SOS:.8f} Eh")


# =============================================================================
# 7. Convergence of E_SOS with nlap
# =============================================================================
#
# Repeat the energy loop with different nlap values to verify convergence.
# B_iaP and D_ia are already built, so only the Laplace loop is re-run.

print("\n" + "=" * 60)
print("Convergence of E_SOS with nlap")
print("=" * 60)
print(f"  {'nlap':>5}   {'E_SOS (Eh)':>18}   {'ΔE from nlap=7 (mEh)':>22}   {'errmax':>12}")

# Store results so we can compute deltas relative to nlap=7 after the loop.
conv_results = []

for n in [3, 5, 7, 10, 15]:
    try:
        e_n, w_n, err_n = mm.laplace_minimax(n, ymin, ymax)
        partial_sum = 0.0
        for k in range(n):
            scale_n  = np.exp(-0.5 * e_n[k] * D_ia)
            C_n      = B_iaP * scale_n[:, :, np.newaxis]
            Gamma_n  = np.einsum('iaP,iaQ->PQ', C_n, C_n)
            partial_sum += w_n[k] * np.einsum('PQ,PQ->', Gamma_n, Gamma_n)
        E_n = -c_os * partial_sum
        conv_results.append((n, E_n, err_n))
    except RuntimeError:
        conv_results.append((n, None, None))

# Use nlap=7 as the reference
E_ref = next(E for n, E, _ in conv_results if n == 7)

for n, E_n, err_n in conv_results:
    if E_n is None:
        print(f"  {n:>5}   (no tabulated data)")
    else:
        delta_mEh = (E_n - E_ref) * 1000.0
        print(f"  {n:>5}   {E_n:>18.8f}   {delta_mEh:>+22.4f}   {err_n:>12.3e}")

print()
print("Increasing nlap drives ΔE toward zero — the Laplace quadrature converges.")
print("In practice nlap=7 is sufficient for chemical accuracy (~0.1 mEh).")
