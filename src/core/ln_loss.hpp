#ifndef MINIMAX_CPPPY_LN_LOSS_HPP
#define MINIMAX_CPPPY_LN_LOSS_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "dd128.hpp"
#include "quadrature.hpp"
#include "expint.hpp"

namespace minimax_cpppy {
namespace detail {

// integer power for DD (p >= 0)
inline DD ddPow(const DD& base, int p) {
    DD result(1.0, 0.0), b = base;
    while (p > 0) {
        if (p & 1) result = result * b;
        p >>= 1;
        if (p) b = DD::ddSquare(b);
    }
    return result;
}

// |x|^p for DD (p >= 0)
inline DD ddPowAbs(const DD& x, int p) {
    DD ax = (x.hi < 0.0) ? -x : x;   // |x|; hi carries the sign
    return ddPow(ax, p);
}

// |x|^p for DD with real exponent p, via exp(p·log|x|).
// At x==0 returns 0 (the only callers use p>0 at zeros; p<0 nodes never land
// on a zero because eta's zeros sit on panel boundaries, off the GL nodes).
// ponytail: no |x|==0 guard for p<0 — split rule keeps nodes off the kinks.
inline DD ddAbsPowReal(const DD& x, double p) {
    DD ax = (x.hi < 0.0) ? -x : x;   // |x|; hi carries the sign
    if (ax.hi == 0.0) return DD(0.0, 0.0);
    return DD::ddExp(DD::ddLog(ax) * DD(p));
}

// L_n = ∫_0^{lnR} |eta(t)|^n e^t dt on a fixed rule, with optional gradient/Hessian.
// The e^t is the x-space Jacobian (dx = e^t dt): this is ∫_1^R |eta|^n dx, the
// L_n error of the 1/x approximation in x-space, not the log-weighted t-space one.
// Mirrors evalLnLoss with eta^{n-1} -> |eta|^{n-1}sign(eta), eta^{n-2} -> |eta|^{n-2}.
// n is a real exponent >= 1 (odd integers and arbitrary doubles); for non-integer
// n the |eta|^n integrand has the same kinks at eta's zeros, so callers must use
// the split rule. The Hessian factor |eta|^{n-2} is singular at zeros for n<2
// (integrable, but conditioning worsens). Caller passes hessAW = nullptr for the
// n == 1 case, where no classical Hessian exists.
inline DD evalLnLossAbs(const QuadRule& rule, int nlap, double n,
                        const DD* a, const DD* w, DD* gradA, DD* gradW,
                        DD* hessAW = nullptr) {
    DD L(0.0, 0.0);
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0, 0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0, 0.0);
    const int dim = 2 * nlap;
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0, 0.0);

    const DD nDD(n);
    const DD nnm1(n * (n - 1.0));
    std::vector<DD> bk(nlap), va(nlap), vw(nlap);

    const size_t N = rule.t.size();
    for (size_t i = 0; i < N; ++i) {
        const DD t(rule.t[i]);
        const DD om(rule.w[i]);
        const DD x   = DD::ddExp(t);
        const DD emt = DD::ddExp(-t);
        DD eta = emt;
        for (int k = 0; k < nlap; ++k) {
            bk[k] = DD::ddExp(-(a[k] * x));
            eta = eta - w[k] * bk[k];
        }
        const DD s(eta.hi < 0.0 ? -1.0 : 1.0);
        const DD gradFac = ddAbsPowReal(eta, n - 1.0) * s;   // |eta|^{n-1} sign(eta)
        const DD valFac  = gradFac * eta;              // |eta|^n
        const DD omx = om * x;                         // dx = e^t dt Jacobian (x-space measure)
        L = L + omx * valFac;

        if (gradA || gradW || hessAW) {
            const DD pref = omx * nDD * gradFac;        // n |eta|^{n-1} sign(eta)
            for (int k = 0; k < nlap; ++k) {
                va[k] = w[k] * (x * bk[k]);
                vw[k] = -bk[k];
                if (gradW) gradW[k] = gradW[k] + pref * vw[k];
                if (gradA) gradA[k] = gradA[k] + pref * va[k];
            }
            if (hessAW) {                              // requires n >= 2
                const DD hessFac = ddAbsPowReal(eta, n - 2.0);   // |eta|^{n-2}
                const DD P2 = omx * nnm1 * hessFac;
                for (int k = 0; k < nlap; ++k) {
                    const DD P2vak = P2 * va[k];
                    const DD P2vwk = P2 * vw[k];
                    for (int l = 0; l < nlap; ++l) {
                        hessAW[k + l * dim] = hessAW[k + l * dim] + P2vak * va[l];
                        DD aw = P2vak * vw[l];
                        hessAW[k + (nlap + l) * dim] = hessAW[k + (nlap + l) * dim] + aw;
                        hessAW[(nlap + l) + k * dim] = hessAW[(nlap + l) + k * dim] + aw;
                        hessAW[(nlap + k) + (nlap + l) * dim] =
                            hessAW[(nlap + k) + (nlap + l) * dim] + P2vwk * vw[l];
                    }
                    const DD x2bk = x * (x * bk[k]);
                    hessAW[k + k * dim] = hessAW[k + k * dim] - pref * (w[k] * x2bk);
                    DD awd = pref * (x * bk[k]);
                    hessAW[k + (nlap + k) * dim] = hessAW[k + (nlap + k) * dim] + awd;
                    hessAW[(nlap + k) + k * dim] = hessAW[(nlap + k) + k * dim] + awd;
                }
            }
        }
    }
    return L;
}

// Smoothed-L1 engine: ∫ sqrt(eta^2 + eps^2) e^t dt, with optional gradient/Hessian.
// As eps -> 0 this approaches ∫|eta| (the n=1 loss). The Hessian
//   H = ∫ [ (eps^2/r^3) eta_i eta_j + (eta/r) eta_ij ] dt,  r = sqrt(eta^2+eps^2)
// has a PSD leading term, so it is well-conditioned for eps>0 and Newton
// converges where the non-smooth |eta| Hessian does not exist. Requires eps>0.
inline DD evalL1LossSmoothed(const QuadRule& rule, int nlap, const DD& eps,
                             const DD* a, const DD* w, DD* gradA, DD* gradW,
                             DD* hessAW = nullptr) {
    DD L(0.0, 0.0);
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0, 0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0, 0.0);
    const int dim = 2 * nlap;
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0, 0.0);

    const DD eps2 = eps * eps;
    std::vector<DD> bk(nlap), va(nlap), vw(nlap);

    const size_t N = rule.t.size();
    for (size_t i = 0; i < N; ++i) {
        const DD t(rule.t[i]);
        const DD om(rule.w[i]);
        const DD x   = DD::ddExp(t);
        const DD emt = DD::ddExp(-t);
        DD eta = emt;
        for (int k = 0; k < nlap; ++k) {
            bk[k] = DD::ddExp(-(a[k] * x));
            eta = eta - w[k] * bk[k];
        }
        const DD r    = DD::ddSquareRoot(eta * eta + eps2);
        const DD invr = DD(1.0) / r;
        const DD omx = om * x;                          // dx = e^t dt Jacobian (x-space measure)
        L = L + omx * r;

        if (gradA || gradW || hessAW) {
            const DD pref = omx * (eta * invr);              // omx * eta / r
            for (int k = 0; k < nlap; ++k) {
                va[k] = w[k] * (x * bk[k]);
                vw[k] = -bk[k];
                if (gradW) gradW[k] = gradW[k] + pref * vw[k];
                if (gradA) gradA[k] = gradA[k] + pref * va[k];
            }
            if (hessAW) {
                const DD P2 = omx * eps2 * (invr * invr * invr);   // omx * eps^2 / r^3
                for (int k = 0; k < nlap; ++k) {
                    const DD P2vak = P2 * va[k];
                    const DD P2vwk = P2 * vw[k];
                    for (int l = 0; l < nlap; ++l) {
                        hessAW[k + l * dim] = hessAW[k + l * dim] + P2vak * va[l];
                        DD aw = P2vak * vw[l];
                        hessAW[k + (nlap + l) * dim] = hessAW[k + (nlap + l) * dim] + aw;
                        hessAW[(nlap + l) + k * dim] = hessAW[(nlap + l) + k * dim] + aw;
                        hessAW[(nlap + k) + (nlap + l) * dim] =
                            hessAW[(nlap + k) + (nlap + l) * dim] + P2vwk * vw[l];
                    }
                    const DD x2bk = x * (x * bk[k]);
                    hessAW[k + k * dim] = hessAW[k + k * dim] - pref * (w[k] * x2bk);
                    DD awd = pref * (x * bk[k]);
                    hessAW[k + (nlap + k) * dim] = hessAW[k + (nlap + k) * dim] + awd;
                    hessAW[(nlap + k) + k * dim] = hessAW[(nlap + k) + k * dim] + awd;
                }
            }
        }
    }
    return L;
}

// Smoothed |eta|^n engine: ∫ (eta^2+eps^2)^(n/2) e^t dt, with optional gradient/Hessian.
// Generalizes evalL1LossSmoothed (the n=1 case) to any real n>=1: as eps -> 0
// this approaches ∫|eta|^n (evalLnLossAbs). r = sqrt(eta^2+eps^2) is always > 0
// for eps>0, so unlike evalLnLossAbs no sign(eta) or split rule is needed --
// the integrand is analytic in t everywhere, including at eta's zeros.
//   dφ/deta   = n eta r^{n-2}
//   d²φ/deta² = n r^{n-4} [(n-1)eta^2 + eps^2]
// (setting n=1 recovers evalL1LossSmoothed's eta/r and eps^2/r^3 exactly).
inline DD evalLnLossAbsSmoothed(const QuadRule& rule, int nlap, double n, const DD& eps,
                                const DD* a, const DD* w, DD* gradA, DD* gradW,
                                DD* hessAW = nullptr) {
    DD L(0.0, 0.0);
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0, 0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0, 0.0);
    const int dim = 2 * nlap;
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0, 0.0);

    const DD eps2 = eps * eps;
    const DD nDD(n);
    const DD nm1(n - 1.0);
    std::vector<DD> bk(nlap), va(nlap), vw(nlap);

    const size_t N = rule.t.size();
    for (size_t i = 0; i < N; ++i) {
        const DD t(rule.t[i]);
        const DD om(rule.w[i]);
        const DD x   = DD::ddExp(t);
        const DD emt = DD::ddExp(-t);
        DD eta = emt;
        for (int k = 0; k < nlap; ++k) {
            bk[k] = DD::ddExp(-(a[k] * x));
            eta = eta - w[k] * bk[k];
        }
        const DD eta2 = eta * eta;
        const DD r2   = eta2 + eps2;
        const DD r    = DD::ddSquareRoot(r2);
        const DD logr = DD::ddLog(r);
        const DD rNm2 = DD::ddExp(logr * DD(n - 2.0));   // r^{n-2}
        const DD omx = om * x;                           // dx = e^t dt Jacobian (x-space measure)
        L = L + omx * r2 * rNm2;                          // r^n = r^2 * r^{n-2}

        if (gradA || gradW || hessAW) {
            const DD pref = omx * nDD * eta * rNm2;       // omx * n * eta * r^{n-2}
            for (int k = 0; k < nlap; ++k) {
                va[k] = w[k] * (x * bk[k]);
                vw[k] = -bk[k];
                if (gradW) gradW[k] = gradW[k] + pref * vw[k];
                if (gradA) gradA[k] = gradA[k] + pref * va[k];
            }
            if (hessAW) {
                const DD rNm4 = rNm2 / r2;                // r^{n-4}
                const DD P2 = omx * nDD * rNm4 * (nm1 * eta2 + eps2);
                for (int k = 0; k < nlap; ++k) {
                    const DD P2vak = P2 * va[k];
                    const DD P2vwk = P2 * vw[k];
                    for (int l = 0; l < nlap; ++l) {
                        hessAW[k + l * dim] = hessAW[k + l * dim] + P2vak * va[l];
                        DD aw = P2vak * vw[l];
                        hessAW[k + (nlap + l) * dim] = hessAW[k + (nlap + l) * dim] + aw;
                        hessAW[(nlap + l) + k * dim] = hessAW[(nlap + l) + k * dim] + aw;
                        hessAW[(nlap + k) + (nlap + l) * dim] =
                            hessAW[(nlap + k) + (nlap + l) * dim] + P2vwk * vw[l];
                    }
                    const DD x2bk = x * (x * bk[k]);
                    hessAW[k + k * dim] = hessAW[k + k * dim] - pref * (w[k] * x2bk);
                    DD awd = pref * (x * bk[k]);
                    hessAW[k + (nlap + k) * dim] = hessAW[k + (nlap + k) * dim] + awd;
                    hessAW[(nlap + k) + k * dim] = hessAW[(nlap + k) + k * dim] + awd;
                }
            }
        }
    }
    return L;
}

// eta(t) in DD at a single node.
inline DD etaAt(double t, int nlap, const DD* a, const DD* w) {
    const DD x = DD::ddExp(DD(t));
    DD s = DD::ddExp(DD(-t));
    for (int k = 0; k < nlap; ++k) s = s - w[k] * DD::ddExp(-(a[k] * x));
    return s;
}

// eta(t) and eta'(t) in DD at a single node, sharing the e^{-a_k e^t} terms.
// eta'(t) = -e^{-t} + Sum_k w_k a_k e^t e^{-a_k e^t}.
inline void etaAndPrimeAt(double t, int nlap, const DD* a, const DD* w,
                          DD& eta, DD& etaPrime) {
    const DD x = DD::ddExp(DD(t));
    const DD emt = DD::ddExp(DD(-t));
    eta = emt;
    etaPrime = -emt;
    for (int k = 0; k < nlap; ++k) {
        const DD bk = DD::ddExp(-(a[k] * x));
        eta = eta - w[k] * bk;
        etaPrime = etaPrime + (w[k] * a[k]) * x * bk;
    }
}

// Refine a bracketed root of eta on [lo,hi] (eta(lo) has sign flo<0, eta(hi) the
// opposite) via safeguarded Newton in DD precision: bisection-narrows the bracket
// every step (guaranteed convergence) while proposing the next point via Newton
// (quadratic convergence) whenever the step stays inside the current bracket.
inline double refineEtaZero(double lo, double hi, double flo,
                            int nlap, const DD* a, const DD* w) {
    double t = 0.5 * (lo + hi);
    for (int it = 0; it < 60; ++it) {
        DD f, fp;
        etaAndPrimeAt(t, nlap, a, w, f, fp);
        if ((f.hi < 0.0) == (flo < 0.0)) { lo = t; flo = f.hi; } else { hi = t; }
        if (f.hi == 0.0 || hi - lo < 1e-15 * (std::fabs(t) + 1.0)) return t;
        double tn = (fp.hi != 0.0) ? t - (f / fp).hi : 0.5 * (lo + hi);
        if (!(tn > lo && tn < hi)) tn = 0.5 * (lo + hi);   // Newton escaped -> bisect
        t = tn;
    }
    return t;
}

// Zeros of eta on (0, lnR): fine uniform scan for sign changes, then safeguarded
// Newton-bisection refinement (both in DD precision -- eta suffers catastrophic
// cancellation in plain double for large nlap / small-to-moderate R, so the scan
// itself must stay in DD; only the refinement algorithm was the cheap-to-improve
// part, see docs/ideas/ln_quad_scan_findings.md).
// ponytail: uniform scan at 400*nlap points reliably brackets the ~2*nlap zeros;
// upgrade to extremum-bracketed search only if a pathological case is found.
inline std::vector<double> findEtaZeros(double R, int nlap, const DD* a, const DD* w) {
    const double b = std::log(R);
    const int M = std::max(2000, 400 * nlap);
    std::vector<double> zeros;
    double tprev = 0.0, fprev = etaAt(tprev, nlap, a, w).hi;
    for (int i = 1; i <= M; ++i) {
        const double tcur = b * static_cast<double>(i) / M;
        const double fcur = etaAt(tcur, nlap, a, w).hi;
        if (fprev != 0.0 && (fprev < 0.0) != (fcur < 0.0)) {
            zeros.push_back(refineEtaZero(tprev, tcur, fprev, nlap, a, w));
        }
        tprev = tcur; fprev = fcur;
    }
    return zeros;
}

// Refines each zero in `prevZeros` toward the corresponding zero of the CURRENT
// eta(a,w) via unconstrained DD Newton steps, exploiting that consecutive outer
// Newton iterations change (a,w) by a small correction so zeros drift only
// slightly. Each refinement is trust-region-limited to half the gap to its
// nearest neighbour in `prevZeros` and residual-checked after convergence; any
// refinement that escapes its trust region, fails to converge in 20 steps, or
// leaves a residual above 1e-10, or a final zero set that isn't still strictly
// increasing, discards ALL refinements and falls back to the full DD scan
// (findEtaZeros) -- this can only be slower than the scan, never wrong.
inline std::vector<double> findEtaZerosWarm(double R, int nlap, const DD* a, const DD* w,
                                            const std::vector<double>& prevZeros) {
    if (prevZeros.empty()) return findEtaZeros(R, nlap, a, w);
    const double b = std::log(R);
    const size_t n = prevZeros.size();

    std::vector<double> zeros(n);
    for (size_t i = 0; i < n; ++i) {
        const double t0 = prevZeros[i];
        const double leftGap  = (i == 0)     ? t0 : (t0 - prevZeros[i - 1]);
        const double rightGap = (i + 1 == n) ? (b - t0) : (prevZeros[i + 1] - t0);
        const double maxJump = 0.5 * std::min(leftGap, rightGap);

        double t = t0;
        bool converged = false;
        for (int it = 0; it < 20; ++it) {
            DD f, fp;
            etaAndPrimeAt(t, nlap, a, w, f, fp);
            if (fp.hi == 0.0) break;
            const double tn = t - (f / fp).hi;
            if (!(tn > 0.0 && tn < b) || std::fabs(tn - t0) > maxJump) break;
            const bool small = std::fabs(tn - t) < 1e-15 * (std::fabs(tn) + 1.0);
            t = tn;
            if (small) { converged = true; break; }
        }
        if (!converged || std::fabs(etaAt(t, nlap, a, w).hi) > 1e-10)
            return findEtaZeros(R, nlap, a, w);   // trust-region/residual failure -> full scan
        zeros[i] = t;
    }
    for (size_t i = 1; i < n; ++i)
        if (zeros[i] <= zeros[i - 1]) return findEtaZeros(R, nlap, a, w);  // collapsed -> full scan

    // Completeness check: the refinements above only ever move the n TRACKED
    // zeros -- they have no way to notice that eta's true zero COUNT changed
    // (e.g. a large outer-Newton step shifts (a,w) enough that a new zero pair
    // appears near the low-t end). Resample eta's sign on a coarse grid (~10x
    // cheaper than findEtaZeros's full scan) and compare the sign-change count
    // to n. If a new zero appeared, the coarse count is > n (caught here). If
    // two tracked zeros now sit inside one coarse cell, the coarse count is
    // < n -- only a spurious full-scan fallback, still correct. Either way this
    // preserves the "never wrong, only slower" contract for every topology
    // change the coarse grid resolves.
    // ponytail: a newly-appeared zero PAIR closer together than the coarse grid
    // spacing b/M2 can still hide (count unchanged, no fallback); such pairs
    // contribute negligibly to the objective, and Debug mode's own alternation
    // cross-check in evalL1LossAnalytic remains a backstop. Upgrade to a finer
    // grid only if a real case is found.
    const int M2 = std::max(200, 40 * nlap);
    int coarseCount = 0;
    double ctprev = 0.0, cfprev = etaAt(ctprev, nlap, a, w).hi;
    for (int i = 1; i <= M2; ++i) {
        const double ctcur = b * static_cast<double>(i) / M2;
        const double cfcur = etaAt(ctcur, nlap, a, w).hi;
        if (cfprev != 0.0 && (cfprev < 0.0) != (cfcur < 0.0)) ++coarseCount;
        ctprev = ctcur; cfprev = cfcur;
    }
    if (coarseCount != static_cast<int>(n))
        return findEtaZeros(R, nlap, a, w);   // zero-count mismatch -> full scan

    return zeros;
}

// Composite GL on [0, lnR] split at eta's zeros (kinks land on panel
// boundaries), doubling total panels until |eta|^n is stable to 1e-12.
// Takes precomputed zeros so callers that already have them (e.g. the outer
// Newton loop in solveLnAbs) don't pay for findEtaZeros twice.
inline QuadRule buildSplitRule(double R, int nlap, double n, const DD* a, const DD* w,
                               const std::vector<double>& zeros) {
    const double b = std::log(R);
    std::vector<double> brk;
    brk.push_back(0.0);
    for (double z : zeros) brk.push_back(z);
    brk.push_back(b);

    const int basePanels = 64 > 8 * nlap ? 64 : 8 * nlap;
    auto build = [&](int mult) -> QuadRule {
        QuadRule rule;
        for (size_t s = 0; s + 1 < brk.size(); ++s) {
            const double lo = brk[s], hi = brk[s + 1];
            int seg = static_cast<int>(std::ceil((hi - lo) / b * basePanels * mult));
            if (seg < 1) seg = 1;
            QuadRule r = compositeGaussLegendre(lo, hi, seg);
            rule.t.insert(rule.t.end(), r.t.begin(), r.t.end());
            rule.w.insert(rule.w.end(), r.w.begin(), r.w.end());
        }
        return rule;
    };

    int mult = 1;
    QuadRule rule = build(mult);
    DD prev = evalLnLossAbs(rule, nlap, n, a, w, nullptr, nullptr);
    for (int it = 0; it < 6; ++it) {
        mult *= 2;
        QuadRule next = build(mult);
        DD cur = evalLnLossAbs(next, nlap, n, a, w, nullptr, nullptr);
        double ref = std::abs(prev.hi) > 1e-300 ? std::abs(prev.hi) : 1.0;
        bool stable = std::abs(cur.hi - prev.hi) / ref < 1e-12;
        rule = std::move(next); prev = cur;
        if (stable) break;
    }
    return rule;
}

// Convenience overload: finds the zeros itself. Prefer the explicit-zeros
// overload above in hot paths that already have them.
inline QuadRule buildSplitRule(double R, int nlap, double n, const DD* a, const DD* w) {
    return buildSplitRule(R, nlap, n, a, w, findEtaZeros(R, nlap, a, w));
}

// Elementary integrals on [1, R] for c > 0 (see plan/spec):
//   H = ∫ e^{-cx} dx, K = ∫ x e^{-cx} dx, M = ∫ x^2 e^{-cx} dx.
// Computed together since they share ec = e^{-c}, ecR = e^{-cR}.
struct L2Elem { DD H, K, M; };
inline L2Elem l2Elem(const DD& c, double R) {
    const DD ec  = DD::ddExp(-c);
    const DD ecR = DD::ddExp(-(c * DD(R)));
    const DD cR  = c * DD(R);
    const DD invc = DD(1.0) / c;
    L2Elem e;
    e.H = (ec - ecR) * invc;
    e.K = (ec * (DD(1.0) + c) - ecR * (DD(1.0) + cR)) * (invc * invc);
    e.M = (ec * (DD(2.0) + DD(2.0) * c + c * c)
           - ecR * (DD(2.0) + DD(2.0) * cR + cR * cR)) * (invc * invc * invc);
    return e;
}

// Analytic L2 loss L = ∫_1^R η^2 dx on [1, R] with η = 1/x - Σ_k w_k e^{-a_k x}.
// Closed form: no quadrature, no zero-finding. Value and w-gradient use the
// exponential integral E1 (via G(c) = E1(c) - E1(cR)); a-gradient and Hessian
// are elementary. gradA/gradW (length nlap) receive ∂L/∂a_k, ∂L/∂w_k when
// non-null. hessAW (2*nlap square, column-major, a_k->k, w_k->nlap+k) is filled
// only when non-null. Requires a_k > 0, R > 1.
inline DD evalL2LossAnalytic(int nlap, double R, const DD* a, const DD* w,
                             DD* gradA, DD* gradW, DD* hessAW = nullptr) {
    const int dim = 2 * nlap;
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0, 0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0, 0.0);
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0, 0.0);

    const DD Rinv = DD(1.0) / DD(R);   // DD reciprocal (no plain-double intermediate)
    const DD two(2.0);

    // Single-argument pieces: G(a_k) (needs E1), H(a_k), K(a_k).
    std::vector<DD> G(nlap), Hk(nlap), Kk(nlap);
    for (int k = 0; k < nlap; ++k) {
        const DD ak = a[k];
        G[k]  = ddExpInt1(ak) - ddExpInt1(ak * DD(R));   // ∫ e^{-a_k x}/x dx
        L2Elem e = l2Elem(ak, R);
        Hk[k] = e.H;
        Kk[k] = e.K;
    }

    // Value: (1 - 1/R) - 2 Σ w_k G(a_k) + Σ_{k,l} w_k w_l H(a_k+a_l).
    DD L = DD(1.0) - Rinv;
    for (int k = 0; k < nlap; ++k) L = L - two * w[k] * G[k];
    for (int k = 0; k < nlap; ++k)
        for (int l = 0; l < nlap; ++l) {
            L2Elem e = l2Elem(a[k] + a[l], R);
            L = L + w[k] * w[l] * e.H;
        }

    if (gradA || gradW) {
        // ∂L/∂w_k = -2 G(a_k) + 2 Σ_l w_l H(a_k+a_l)
        // ∂L/∂a_k =  2 w_k [ H(a_k) - Σ_l w_l K(a_k+a_l) ]   (= 2 w_k S_k)
        for (int k = 0; k < nlap; ++k) {
            DD sumH(0.0, 0.0), sumK(0.0, 0.0);
            for (int l = 0; l < nlap; ++l) {
                L2Elem e = l2Elem(a[k] + a[l], R);
                sumH = sumH + w[l] * e.H;
                sumK = sumK + w[l] * e.K;
            }
            if (gradW) gradW[k] = -two * G[k] + two * sumH;
            if (gradA) gradA[k] = two * w[k] * (Hk[k] - sumK);
        }
    }
    if (hessAW) {
        // S_k = H(a_k) - Σ_l w_l K(a_k+a_l);  T_k = K(a_k) - Σ_l w_l M(a_k+a_l).
        std::vector<DD> S(nlap), T(nlap);
        for (int k = 0; k < nlap; ++k) {
            DD sumK(0.0, 0.0), sumM(0.0, 0.0);
            for (int l = 0; l < nlap; ++l) {
                L2Elem e = l2Elem(a[k] + a[l], R);
                sumK = sumK + w[l] * e.K;
                sumM = sumM + w[l] * e.M;
            }
            S[k] = Hk[k] - sumK;
            T[k] = Kk[k] - sumM;
        }
        for (int k = 0; k < nlap; ++k) {
            for (int l = 0; l < nlap; ++l) {
                L2Elem e = l2Elem(a[k] + a[l], R);
                // ww block: 2 H(a_k+a_l)
                hessAW[(nlap + k) + (nlap + l) * dim] = two * e.H;
                // aa block: 2 w_k w_l M(a_k+a_l), minus diagonal 2 w_k T_k
                DD aa = two * w[k] * w[l] * e.M;
                if (k == l) aa = aa - two * w[k] * T[k];
                hessAW[k + l * dim] = aa;
                // aw block (and symmetric wa): -2 w_k K(a_k+a_l), plus diag 2 S_k
                DD aw = -two * w[k] * e.K;
                if (k == l) aw = aw + two * S[k];
                hessAW[k + (nlap + l) * dim] = aw;
                hessAW[(nlap + l) + k * dim] = aw;
            }
        }
    }
    return L;
}

// Analytic L1 loss L = ∫_1^R |eta| dx on [1,R], evaluated in t-space (t=ln x)
// with eta's zeros passed in as t-values on (0, lnR), strictly increasing.
// Elementary closed form: no quadrature, no E1 (the |eta| integrand's
// antiderivative is fully elementary). Value, gradient, and Hessian are exact
// to DD. Define the x-space-measure integrand psi(t) = eta(t) e^t; since e^t>0,
// |eta| dx = |psi| dt and psi shares eta's zeros, so L = ∫_0^{lnR} |psi| dt.
// psi has constant sign s_j on each panel (t_j, t_{j+1}) and the signs alternate.
//
// gradA/gradW (length nlap) and hessAW (2*nlap square, column-major, a_k->k,
// w_k->nlap+k) are filled when non-null; same contract as evalLnLossAbs. The
// Hessian is a diagonal smooth part plus one PSD rank-1 correction per interior
// zero (the zeros move with (a,w)). Requires a_k>0, R>1. Sets *ok=false (when
// ok!=null) on a near-tangent zero (|eta_t(t_j)| tiny), where the rank-1
// coefficient blows up, so the caller can fall back to the smoothed solver.
inline DD evalL1LossAnalytic(int nlap, double R,
                             const DD* a, const DD* w,
                             DD* gradA, DD* gradW, DD* hessAW,
                             const std::vector<double>& zeros,
                             bool* ok = nullptr) {
    const int dim = 2 * nlap;
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0, 0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0, 0.0);
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0, 0.0);
    if (ok) *ok = true;

    const double b = std::log(R);
    const int m = static_cast<int>(zeros.size());

    // Breakpoints t_0=0, t_1..t_m = zeros, t_{m+1}=b. Share x_i=e^{t_i} and
    // bk_i[k]=e^{-a_k x_i} across value/grad/Hessian at each breakpoint.
    std::vector<double> tb(m + 2);
    tb[0] = 0.0;
    for (int i = 0; i < m; ++i) tb[i + 1] = zeros[i];
    tb[m + 1] = b;
    std::vector<DD> xb(m + 2);
    std::vector<std::vector<DD>> bkb(m + 2, std::vector<DD>(nlap));
    for (int i = 0; i < m + 2; ++i) {
        xb[i] = DD::ddExp(DD(tb[i]));
        for (int k = 0; k < nlap; ++k) bkb[i][k] = DD::ddExp(-(a[k] * xb[i]));
    }

    // Panel sign bootstrap: s_0 = sign(psi(0+)) = sign(eta(0)) = sign(1 - Σ w_k e^{-a_k}).
    DD eta0 = DD(1.0);
    for (int k = 0; k < nlap; ++k) eta0 = eta0 - w[k] * bkb[0][k];
    const double s0 = (eta0.hi < 0.0) ? -1.0 : 1.0;

    // F̃(t) = t + Σ_k (w_k/a_k) e^{-a_k e^t};  L = Σ_j s_j [F̃(t_{j+1}) - F̃(t_j)].
    // Gradient x-primitives (indefinite, evaluated at panel bounds):
    //   P0(c;x) = -e^{-cx}/c        (primitive of e^{-cx})
    //   P1(c;x) = -e^{-cx}(1+cx)/c^2 (primitive of x e^{-cx})
    // ∂L/∂w_k = -Σ_j s_j [P0(a_k)] ; ∂L/∂a_k = +Σ_j s_j w_k [P1(a_k)].
    DD L(0.0, 0.0);
    for (int j = 0; j <= m; ++j) {
        const DD sD((j % 2 == 0) ? s0 : -s0);
        DD Fr(tb[j + 1]), Fl(tb[j]);
        for (int k = 0; k < nlap; ++k) {
            const DD woa = w[k] / a[k];
            Fr = Fr + woa * bkb[j + 1][k];
            Fl = Fl + woa * bkb[j][k];
        }
        L = L + sD * (Fr - Fl);

        if (gradA || gradW || hessAW) {
            for (int k = 0; k < nlap; ++k) {
                const DD ak = a[k];
                const DD invc = DD(1.0) / ak;
                const DD invc2 = invc * invc;
                const DD br = bkb[j + 1][k], bl = bkb[j][k];
                const DD cxr = ak * xb[j + 1], cxl = ak * xb[j];
                const DD dP0 = (-(br) * invc) - (-(bl) * invc);
                const DD dP1 = (-(br * (DD(1.0) + cxr)) * invc2)
                             - (-(bl * (DD(1.0) + cxl)) * invc2);
                if (gradW) gradW[k] = gradW[k] - sD * dP0;
                if (gradA) gradA[k] = gradA[k] + sD * (w[k] * dP1);
                if (hessAW) {
                    // P2(c;x) = -e^{-cx}(2+2cx+c^2 x^2)/c^3 (primitive of x^2 e^{-cx}).
                    const DD invc3 = invc2 * invc;
                    const DD dP2 = (-(br * (DD(2.0) + DD(2.0) * cxr + cxr * cxr)) * invc3)
                                 - (-(bl * (DD(2.0) + DD(2.0) * cxl + cxl * cxl)) * invc3);
                    // [aa]_kk -= s_j w_k [P2] ; [aw]_kk += s_j [P1] ; [ww]=0.
                    hessAW[k + k * dim] = hessAW[k + k * dim] - sD * (w[k] * dP2);
                    const DD aw = sD * dP1;
                    hessAW[k + (nlap + k) * dim] = hessAW[k + (nlap + k) * dim] + aw;
                    hessAW[(nlap + k) + k * dim] = hessAW[(nlap + k) + k * dim] + aw;
                }
            }
        }
    }

    // Moving-zero PSD rank-1 corrections at each interior zero t_j (j=1..m):
    //   H += c_j g_j g_j^T,  c_j = -2 s_{j-1} x_j / eta_t(t_j),  g_j = eta_θ(t_j).
    // eta_{a_k}(t_j) = w_k x_j b_k^{(j)},  eta_{w_k}(t_j) = -b_k^{(j)}.
    // c_j > 0 for a simple zero (sign(eta_t)= -s_{j-1}); |eta_t| tiny => tangent.
    if (hessAW) {
        // ponytail: absolute tangent tol; the solver's line search + non-converge
        // fallback catch anything this misses, so exact value is not critical.
        const double TANGENT_TOL = 1e-12;
        std::vector<DD> g(dim);
        for (int j = 1; j <= m; ++j) {
            DD eta, etaP;
            etaAndPrimeAt(tb[j], nlap, a, w, eta, etaP);
            if (std::fabs(etaP.hi) < TANGENT_TOL) { if (ok) *ok = false; continue; }
            const double sjm1 = ((j - 1) % 2 == 0) ? s0 : -s0;
            const DD cj = DD(-2.0 * sjm1) * xb[j] / etaP;
            for (int k = 0; k < nlap; ++k) {
                g[k]        = w[k] * (xb[j] * bkb[j][k]);   // eta_{a_k}
                g[nlap + k] = -bkb[j][k];                   // eta_{w_k}
            }
            for (int p = 0; p < dim; ++p) {
                const DD cg = cj * g[p];
                for (int q = 0; q < dim; ++q)
                    hessAW[p + q * dim] = hessAW[p + q * dim] + cg * g[q];
            }
        }
    }

#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    // Cross-check the alternation against sign(psi)=sign(eta) at each panel
    // midpoint (catches a missed/spurious zero in the passed-in list). Skipped
    // when the rank-1 pass already flagged *ok=false: a near-tangent "zero" can
    // be a real extremum rather than a true sign change (eta_t~=0 without
    // eta crossing), which trips this alternation check spuriously; the caller
    // already knows to distrust the result and fall back, so a second (noisier)
    // diagnostic here would be redundant.
    if (!(ok && !*ok)) {
        for (int j = 0; j <= m; ++j) {
            const double tm = 0.5 * (tb[j] + tb[j + 1]);
            const double etaM = etaAt(tm, nlap, a, w).hi;
            const double sj = (j % 2 == 0) ? s0 : -s0;
            if (etaM != 0.0 && ((etaM < 0.0) ? -1.0 : 1.0) != sj)
                throw std::runtime_error(
                    "evalL1LossAnalytic: panel sign alternation violated at panel "
                    + std::to_string(j) + " (missed/spurious zero?)");
        }
    }
#endif
    return L;
}

// L_n = ∫_0^{lnR} eta(t)^n e^t dt on a fixed rule, with optional gradient and Hessian.
// The e^t is the x-space Jacobian (dx = e^t dt); this is ∫_1^R eta^n dx.
// eta(t) = e^{-t} - Σ_k w_k e^{-a_k e^t}.  n must be >= 1 for value/gradient
// (caller passes even n>=2); the Hessian additionally requires n>=2.
//
// gradA/gradW (length nlap) receive ∂L/∂a_k, ∂L/∂w_k when non-null.
// hessAW, when non-null, receives the exact Hessian in (a,w) parameter space as a
// (2*nlap)x(2*nlap) column-major matrix: index a_k -> k, index w_k -> nlap+k.
inline DD evalLnLoss(const QuadRule& rule, int nlap, int n,
                     const DD* a, const DD* w, DD* gradA, DD* gradW,
                     DD* hessAW = nullptr) {
    DD L(0.0, 0.0);
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0, 0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0, 0.0);
    const int dim = 2 * nlap;
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0, 0.0);

    const DD nDD(static_cast<double>(n));
    const DD nnm1(static_cast<double>(n) * (n - 1));  // n(n-1)
    std::vector<DD> bk(nlap);   // e^{-a_k e^t}
    std::vector<DD> va(nlap);   // η_{a_k} = w_k e^t e^{-a_k e^t}
    std::vector<DD> vw(nlap);   // η_{w_k} = -e^{-a_k e^t}

    const size_t N = rule.t.size();
    for (size_t i = 0; i < N; ++i) {
        const DD t(rule.t[i]);
        const DD om(rule.w[i]);
        const DD x   = DD::ddExp(t);
        const DD emt = DD::ddExp(-t);            // e^{-t}
        DD eta = emt;
        for (int k = 0; k < nlap; ++k) {
            bk[k] = DD::ddExp(-(a[k] * x));      // e^{-a_k e^t}
            eta = eta - w[k] * bk[k];
        }
        const DD etaNm1 = ddPow(eta, n - 1);
        const DD etaN   = eta * etaNm1;
        const DD omx = om * x;                   // dx = e^t dt Jacobian (x-space measure)
        L = L + omx * etaN;

        if (gradA || gradW || hessAW) {
            const DD pref = omx * nDD * etaNm1;   // P1 = ω · n · η^{n-1}
            for (int k = 0; k < nlap; ++k) {
                va[k] = w[k] * (x * bk[k]);       // η_{a_k}
                vw[k] = -bk[k];                   // η_{w_k}
                if (gradW) gradW[k] = gradW[k] + pref * vw[k];   // -n ∫ η^{n-1} e^{-a_k e^t}
                if (gradA) gradA[k] = gradA[k] + pref * va[k];   // +n w_k ∫ η^{n-1} e^t e^{-a_k e^t}
            }
            if (hessAW) {
                const DD etaNm2 = ddPow(eta, n - 2);             // requires n>=2
                const DD P2 = omx * nnm1 * etaNm2;                // ω · n(n-1) · η^{n-2}
                for (int k = 0; k < nlap; ++k) {
                    const DD P2vak = P2 * va[k];
                    const DD P2vwk = P2 * vw[k];
                    for (int l = 0; l < nlap; ++l) {
                        // aa block
                        hessAW[k + l * dim] = hessAW[k + l * dim] + P2vak * va[l];
                        // aw block (and its symmetric wa entry)
                        DD aw = P2vak * vw[l];
                        hessAW[k + (nlap + l) * dim] = hessAW[k + (nlap + l) * dim] + aw;
                        hessAW[(nlap + l) + k * dim] = hessAW[(nlap + l) + k * dim] + aw;
                        // ww block
                        hessAW[(nlap + k) + (nlap + l) * dim] =
                            hessAW[(nlap + k) + (nlap + l) * dim] + P2vwk * vw[l];
                    }
                    // diagonal second-partial-of-η contributions (P1 · η_{ij})
                    const DD x2bk = x * (x * bk[k]);
                    hessAW[k + k * dim] = hessAW[k + k * dim] - pref * (w[k] * x2bk); // η_{a_k a_k}=-w_k x^2 b_k
                    DD awd = pref * (x * bk[k]);                                       // η_{a_k w_k}= x b_k
                    hessAW[k + (nlap + k) * dim] = hessAW[k + (nlap + k) * dim] + awd;
                    hessAW[(nlap + k) + k * dim] = hessAW[(nlap + k) + k * dim] + awd;
                }
            }
        }
    }
    return L;
}

// Build the integration rule on [0, lnR], doubling panels until L_n is stable.
inline QuadRule buildLnRule(double R, int nlap, int n, const DD* a, const DD* w) {
    const double b = std::log(R);
    int panels = 64 > 8 * nlap ? 64 : 8 * nlap;
    QuadRule rule = compositeGaussLegendre(0.0, b, panels);
    DD prev = evalLnLoss(rule, nlap, n, a, w, nullptr, nullptr);
    for (int it = 0; it < 6; ++it) {
        panels *= 2;
        QuadRule next = compositeGaussLegendre(0.0, b, panels);
        DD cur = evalLnLoss(next, nlap, n, a, w, nullptr, nullptr);
        double ref = std::abs(prev.hi) > 1e-300 ? std::abs(prev.hi) : 1.0;
        bool stable = std::abs(cur.hi - prev.hi) / ref < 1e-12;
        rule = std::move(next); prev = cur;
        if (stable) break;
    }
    return rule;
}

// Composite GL on [0, lnR] split at the UNREGULARIZED eta's zeros, doubling
// panels until the eps-regularized |eta|^n loss (value AND gradient) is stable.
// For eps>0 the integrand has no true kink (r=sqrt(eta^2+eps^2) never touches
// zero), so a uniform grid is *correct* -- but it's a bad *idea*: the smoothed
// peak is still centered exactly at eta's zero and only ~eps wide, and a
// uuniform grid has to resolve that arbitrarily sharp, localized feature by
// brute-force global refinement (measured: >4M panels needed at eps~1e-6..1e-7
// for some warm-started configurations, even at modest nlap). Splitting at the
// same zeros buildSplitRule uses gets panels already dense at the right place,
// needing far less additional refinement. Unlike buildSplitRule (which checks
// the unregularized L -- fine there, since splitting alone handles the true
// kink), this checks the smoothed loss itself, since for small eps an L-only
// check on a coarse grid can plateau before the gradient has converged.
inline QuadRule buildSplitSmoothedRule(double R, int nlap, double n, const DD& eps,
                                       const DD* a, const DD* w,
                                       const std::vector<double>& zeros) {
    const double b = std::log(R);
    std::vector<double> brk;
    brk.push_back(0.0);
    for (double z : zeros) brk.push_back(z);
    brk.push_back(b);

    const int basePanels = 64 > 8 * nlap ? 64 : 8 * nlap;
    auto build = [&](int mult) -> QuadRule {
        QuadRule rule;
        for (size_t s = 0; s + 1 < brk.size(); ++s) {
            const double lo = brk[s], hi = brk[s + 1];
            int seg = static_cast<int>(std::ceil((hi - lo) / b * basePanels * mult));
            if (seg < 1) seg = 1;
            QuadRule r = compositeGaussLegendre(lo, hi, seg);
            rule.t.insert(rule.t.end(), r.t.begin(), r.t.end());
            rule.w.insert(rule.w.end(), r.w.begin(), r.w.end());
        }
        return rule;
    };

    std::vector<DD> gA(nlap), gW(nlap);
    auto gradInfNorm = [&]() {
        double m = 0.0;
        for (int k = 0; k < nlap; ++k) {
            m = std::max(m, std::fabs(gA[k].hi));
            m = std::max(m, std::fabs(gW[k].hi));
        }
        return m;
    };

    int mult = 1;
    const int maxMult = 1 << 12;   // ~4.2M points at basePanels=64, bounded regardless of nlap
    QuadRule rule = build(mult);
    DD prevL = evalLnLossAbsSmoothed(rule, nlap, n, eps, a, w, gA.data(), gW.data());
    double prevG = gradInfNorm();
    while (mult < maxMult) {
        mult = std::min(mult * 2, maxMult);
        QuadRule next = build(mult);
        DD curL = evalLnLossAbsSmoothed(next, nlap, n, eps, a, w, gA.data(), gW.data());
        double curG = gradInfNorm();
        double refL = std::abs(prevL.hi) > 1e-300 ? std::abs(prevL.hi) : 1.0;
        double refG = prevG > 1e-300 ? prevG : 1.0;
        bool stableL = std::abs(curL.hi - prevL.hi) / refL < 1e-12;
        bool stableG = std::abs(curG - prevG) / refG < 1e-10;
        rule = std::move(next); prevL = curL; prevG = curG;
        if (stableL && stableG) break;
    }
    return rule;
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_LN_LOSS_HPP
