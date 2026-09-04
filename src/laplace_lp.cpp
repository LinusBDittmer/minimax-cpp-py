// src/laplace_lp.cpp
#include "minimax_cpppy/laplace_lp.hpp"
#include "core/algorithm.hpp"   // detail::laplaceMinimax
#include "core/newton.hpp"
#include "core/ln_loss.hpp"
#include "core/log_pretty.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax_cpppy {

// chain rule a_k = e^{p_k}: map (a,w)-space grad/Hess into z=(p,w)-space.
//   grad_p_k = a_k g_{a_k};  grad_w_k = g_{w_k}
//   H_{p_k p_l} = a_k a_l H_{a_k a_l} + δ_kl a_k g_{a_k}
//   H_{p_k w_l} = a_k H_{a_k w_l};   H_{w_k w_l} = H_{w_k w_l}
static void chainRuleToZ(int nlap, int dim, const detail::DD* a,
                         const detail::DD* gA, const detail::DD* gW,
                         const detail::DD* hAW, detail::DD* grad, detail::DD* hess) {
    using detail::DD;
    if (grad) for (int k = 0; k < nlap; ++k) { grad[k] = gA[k] * a[k]; grad[nlap + k] = gW[k]; }
    if (hess) {
        for (int k = 0; k < nlap; ++k) for (int l = 0; l < nlap; ++l) {
            DD pp = (a[k] * a[l]) * hAW[k + l * dim];
            if (k == l) pp = pp + a[k] * gA[k];
            hess[k + l * dim] = pp;
            DD pw = a[k] * hAW[k + (nlap + l) * dim];
            hess[k + (nlap + l) * dim] = pw;
            hess[(nlap + l) + k * dim] = pw;
            hess[(nlap + k) + (nlap + l) * dim] = hAW[(nlap + k) + (nlap + l) * dim];
        }
    }
}

// Solve the |eta|^n (abs) problem from a given warm start z=[ln a, w], using the
// split-rule freeze/re-split outer loop. Returns a NewtonResult whose `converged`
// flag reports success; never throws. Caller decides what to do on failure.
static detail::NewtonResult solveLnAbs(double R, int nlap, double n,
                                       const std::vector<detail::DD>& zInit,
                                       int maxOuter, int maxIter,
                                       int verbose, std::ostream* os) {
    using detail::DD;
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap);
    std::vector<DD> hAW(static_cast<size_t>(dim) * dim);
    detail::QuadRule rule;   // rebuilt each outer iteration

    auto fOdd = [&](const DD* z, DD* grad, DD* hess) -> DD {
        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(z[k]); w[k] = z[nlap + k]; }
        const bool needG = (grad || hess);
        DD* hessArg = hess ? hAW.data() : nullptr;
        DD L = detail::evalLnLossAbs(rule, nlap, n, a.data(), w.data(),
                                     needG ? gA.data() : nullptr,
                                     needG ? gW.data() : nullptr, hessArg);
        if (grad || hess)
            chainRuleToZ(nlap, dim, a.data(), gA.data(), gW.data(),
                         hAW.data(), grad, hess);
        return L;
    };

    detail::NewtonResult nr;
    nr.x = zInit;
    std::vector<DD> zcur = zInit;
    // Zeros at the current zcur, carried forward from the previous outer
    // iteration's post-Newton zero-finding call (same (a,w), so it's the exact
    // same computation -- reusing it skips a whole DD scan every iteration but
    // the first). Empty on the first iteration, forcing a cold full scan.
    std::vector<double> zSet;
    for (int outer = 0; outer < maxOuter; ++outer) {
        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(zcur[k]); w[k] = zcur[nlap + k]; }
        std::vector<double> zUsed =
            zSet.empty() ? detail::findEtaZeros(R, nlap, a.data(), w.data()) : zSet;
        rule = detail::buildSplitRule(R, nlap, n, a.data(), w.data(), zUsed);

        nr = detail::newtonMinimize(dim, fOdd, zcur.data(), maxIter, 1e-10, verbose, os);
        if (!nr.converged) return nr;   // Newton stalled -> report failure
        zcur = nr.x;

        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(zcur[k]); w[k] = zcur[nlap + k]; }
        std::vector<double> zNew = detail::findEtaZerosWarm(R, nlap, a.data(), w.data(), zUsed);
        zSet = zNew;   // reused as next outer iteration's zUsed
        double drift = 1e9;
        if (zUsed.size() == zNew.size()) {
            drift = 0.0;
            for (size_t i = 0; i < zUsed.size(); ++i)
                drift = std::max(drift, std::fabs(zUsed[i] - zNew[i]));
        }
        if (drift < 1e-9) { nr.converged = true; nr.x = zcur; return nr; }
    }
    nr.converged = false;   // zeros kept drifting -> treat as not converged
    return nr;
}

// Solve the |eta|^n loss via eps-regularized smoothing: min integral of
// (eta^2+eps^2)^(n/2), annealing eps from an eps0 hint down toward 0 over up to
// four levels (stopping early once the solution itself stops moving), warm-
// starting each level from the previous. Generalizes the n=1
// smoothed-L1 approach (a PSD-dominant, well-conditioned Hessian for eps>0,
// unlike the near-singular |eta|^{n-2} Hessian at eta's zeros for n<2) to any
// n>=1 -- this replaces the old bisection-in-n continuation, which stacked many
// full solveLnAbs calls (each itself struggling to converge near the singular
// Hessian) and measured 20x-80x+ slower than a well-behaved direct solve even
// at n as close to 2 as 1.15. Rule is split at the unregularized eta's zeros
// (buildSplitSmoothedRule) even though eps>0 removes the true kink: the smoothed
// peak is still centered exactly there and only ~eps wide, and resolving that
// with a uniform grid needs far more panels than starting from split-panel
// density. The final reported norm still uses the exact (split-rule) |eta|^n.
// Throws if a level fails to converge.
static detail::NewtonResult solveLnSmoothed(double R, int nlap, double n,
                                            const std::vector<detail::DD>& z0,
                                            double eps0hint, int verbose, std::ostream* os) {
    using detail::DD;
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap);
    std::vector<DD> hAW(static_cast<size_t>(dim) * dim);
    detail::QuadRule rule;   // rebuilt per eps level
    DD eps(0.0, 0.0);

    auto fSmooth = [&](const DD* z, DD* grad, DD* hess) -> DD {
        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(z[k]); w[k] = z[nlap + k]; }
        const bool needG = (grad || hess);
        DD L = detail::evalLnLossAbsSmoothed(rule, nlap, n, eps, a.data(), w.data(),
                                             needG ? gA.data() : nullptr,
                                             needG ? gW.data() : nullptr,
                                             hess ? hAW.data() : nullptr);
        if (grad || hess)
            chainRuleToZ(nlap, dim, a.data(), gA.data(), gW.data(), hAW.data(), grad, hess);
        return L;
    };

    std::vector<DD> zcur = z0;
    detail::NewtonResult nr;
    const double eps0 = (eps0hint > 0.0) ? eps0hint : 1e-3;
    // 4 levels, not 6: resolving the smoothed peak to eps needs O(1/eps) panels
    // even with split placement, so cost explodes well before eps0/1e5 while the
    // solution itself has already stopped moving (measured: eps0/1e4 and eps0/1e5
    // changed the solution by ~1e-8 relative while costing >80% of total time).
    for (int lvl = 0; lvl < 4; ++lvl) {
        eps = DD(eps0 * std::pow(10.0, -lvl));   // eps0, eps0/10, eps0/100, eps0/1000
        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(zcur[k]); w[k] = zcur[nlap + k]; }
        std::vector<double> zeros = detail::findEtaZeros(R, nlap, a.data(), w.data());
        rule = detail::buildSplitSmoothedRule(R, nlap, n, eps, a.data(), w.data(), zeros);
        nr = detail::newtonMinimize(dim, fSmooth, zcur.data(), 200, 1e-10, verbose, os);
        if (!nr.converged)
            throw std::runtime_error(
                "laplaceLp: smoothed solve failed to converge at n=" + std::to_string(n) +
                " (eps level " + std::to_string(lvl) + ", |grad|_inf=" +
                std::to_string(nr.gradNormInf) + ")");
        // Each level down in eps costs far more (finer panels needed to resolve
        // the ever-narrower smoothed peak) while contributing ever less to the
        // actual solution once it has settled -- stop annealing as soon as the
        // solution itself stops moving between levels, instead of always paying
        // for all 6 (measured: the last two levels can cost >80% of total wall
        // time for a ~1e-8 relative change in the solution).
        if (lvl > 0) {
            double maxChange = 0.0;
            for (int i = 0; i < dim; ++i)
                maxChange = std::max(maxChange, std::fabs((nr.x[i] - zcur[i]).hi));
            zcur = nr.x;
            if (maxChange < 1e-9) break;
        } else {
            zcur = nr.x;
        }
    }
    // Report the TRUE L_n norm (integral of |eta|^n), not the smoothed value.
    for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(zcur[k]); w[k] = zcur[nlap + k]; }
    rule = detail::buildSplitRule(R, nlap, n, a.data(), w.data());
    nr.fval = detail::evalLnLossAbs(rule, nlap, n, a.data(), w.data(), nullptr, nullptr);
    nr.x = zcur;
    return nr;
}

// Solve the exact L1 problem (∫|eta| dx) analytically from warm start z=[ln a, w].
// Single Newton loop: each objective call recomputes eta's zeros (warm-started
// from the previous evaluation) and calls evalL1LossAnalytic for the exact
// value/gradient/Hessian (incl. PSD rank-1 kink corrections at the current
// zeros). No frozen-rule outer loop -- "the rule" is just the zero list,
// recomputed each call, so Newton sees the exact objective and converges
// quadratically. Returns converged=false on a near-tangent zero (evalL1LossAnalytic
// ok==false) or a Newton stall; never throws.
static detail::NewtonResult solveL1Analytic(double R, int nlap,
                                            const std::vector<detail::DD>& z0,
                                            int verbose, std::ostream* os) {
    using detail::DD;
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap);
    std::vector<DD> hAW(static_cast<size_t>(dim) * dim);
    std::vector<double> zSet;   // warm-start zeros, carried across objective calls
    bool degenerate = false;

    auto fL1 = [&](const DD* z, DD* grad, DD* hess) -> DD {
        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(z[k]); w[k] = z[nlap + k]; }
        zSet = detail::findEtaZerosWarm(R, nlap, a.data(), w.data(), zSet);
        const bool needG = (grad || hess);
        bool ok = true;
        DD L = detail::evalL1LossAnalytic(nlap, R, a.data(), w.data(),
                                          needG ? gA.data() : nullptr,
                                          needG ? gW.data() : nullptr,
                                          hess ? hAW.data() : nullptr, zSet, &ok);
        if (!ok) degenerate = true;
        if (grad || hess)
            chainRuleToZ(nlap, dim, a.data(), gA.data(), gW.data(),
                         hAW.data(), grad, hess);
        return L;
    };

    // Defense-in-depth for the "never throws" contract: Change A (the coarse
    // completeness check in findEtaZerosWarm) removes the practical throw path,
    // but if the objective still throws (e.g. a Debug-mode cross-check trips on
    // some case the coarse grid can't resolve), degrade to converged=false so
    // the caller's fallback runs instead of the exception escaping laplaceLp.
    detail::NewtonResult nr;
    try {
        nr = detail::newtonMinimize(dim, fL1, z0.data(), 200, 1e-10, verbose, os);
    } catch (const std::exception&) {
        nr.converged = false;
        return nr;
    }
    if (degenerate) nr.converged = false;   // near-tangent zero -> bail to fallback
    return nr;
}

MinimaxResult laplaceLp(int nlap, double ymin, double ymax, double normP,
                        int verbose, std::ostream& os) {
    using detail::DD;
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (nlap < 1 || nlap > 30)
        throw std::invalid_argument("laplaceLp: nlap out of [1,30]");
    if (ymin <= 0.0 || ymax <= ymin)
        throw std::invalid_argument("laplaceLp: need 0 < ymin < ymax");
    if (normP < 1.0)
        throw std::invalid_argument("laplaceLp: normP must be >= 1");
#endif
    if (verbose >= 1) detail::printRunBanner(os, "laplaceLp", nlap, ymin, ymax);
    const bool isInt   = (normP == std::floor(normP));
    const bool evenInt = isInt && (static_cast<long long>(normP) % 2 == 0);
    if (verbose >= 2 && normP > 20)
        os << "[warn] laplaceLp: normP=" << normP
           << " > 20; eta^p underflows and Newton conditioning worsens; "
              "result is ~indistinguishable from minimax.\n";

    const double R = ymax / ymin;

    // 1. Unbiased minimax warm-start in the normalised [1, R] domain.
    MinimaxResult mm = detail::laplaceMinimax(nlap, 1.0, R, 200, 1e-10, 1e-15,
                                              0.3, 1e-6, 1e-4, verbose,
                                              verbose ? &os : nullptr);

    // 2. Reparametrise: z = [p_k = ln a_k, w_k].
    const int dim = 2 * nlap;
    std::vector<DD> z0(dim);
    for (int k = 0; k < nlap; ++k) {
        z0[k]        = DD::ddLog(DD(mm.expon[k]));
        z0[nlap + k] = DD(mm.weight[k]);
    }

    detail::NewtonResult nr;

    if (normP == 2.0) {
        // ---- n == 2: analytic L2 (closed form + E1); no quadrature ----
        std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap);
        std::vector<DD> hAW(static_cast<size_t>(dim) * dim);
        auto fL2 = [&](const DD* z, DD* grad, DD* hess) -> DD {
            for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(z[k]); w[k] = z[nlap + k]; }
            const bool needG = (grad || hess);
            DD L = detail::evalL2LossAnalytic(nlap, R, a.data(), w.data(),
                                              needG ? gA.data() : nullptr,
                                              needG ? gW.data() : nullptr,
                                              hess ? hAW.data() : nullptr);
            if (grad || hess)
                chainRuleToZ(nlap, dim, a.data(), gA.data(), gW.data(),
                             hAW.data(), grad, hess);
            return L;
        };
        nr = detail::newtonMinimize(dim, fL2, z0.data(), 200, 1e-10,
                                    verbose, verbose ? &os : nullptr);
        if (!nr.converged)
            throw std::runtime_error(
                "laplaceLp: analytic normP=2 Newton failed to converge (|grad|_inf=" +
                std::to_string(nr.gradNormInf) + " after " +
                std::to_string(nr.iters) + " iters)");
    } else if (evenInt) {
        // ---- even integer normP: |eta|^p == eta^p; existing single-solve Newton path ----
        const int ni = static_cast<int>(normP);
        std::vector<DD> a0(nlap), w0(nlap);
        for (int k = 0; k < nlap; ++k) { a0[k] = DD(mm.expon[k]); w0[k] = DD(mm.weight[k]); }
        detail::QuadRule rule = detail::buildLnRule(R, nlap, ni, a0.data(), w0.data());

        std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap);
        std::vector<DD> hAW(static_cast<size_t>(dim) * dim);
        auto fEven = [&](const DD* z, DD* grad, DD* hess) -> DD {
            for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(z[k]); w[k] = z[nlap + k]; }
            const bool needG = (grad || hess);
            DD L = detail::evalLnLoss(rule, nlap, ni, a.data(), w.data(),
                                      needG ? gA.data() : nullptr,
                                      needG ? gW.data() : nullptr,
                                      hess ? hAW.data() : nullptr);
            if (grad || hess)
                chainRuleToZ(nlap, dim, a.data(), gA.data(), gW.data(),
                             hAW.data(), grad, hess);
            return L;
        };
        nr = detail::newtonMinimize(dim, fEven, z0.data(), 200, 1e-10,
                                    verbose, verbose ? &os : nullptr);
        if (!nr.converged)
            throw std::runtime_error(
                "laplaceLp: Newton failed to converge (|grad|_inf=" +
                std::to_string(nr.gradNormInf) + " after " +
                std::to_string(nr.iters) + " iters)");
    } else if (normP == 1.0) {
        // ---- normP == 1: analytic L1 (elementary closed form + DD zero-finding) ----
        // Newton with the exact rank-1-corrected Hessian. Fall back to the
        // eps-smoothed solver on a near-tangent zero or a Newton stall.
        std::ostream* vos = verbose ? &os : nullptr;
        nr = solveL1Analytic(R, nlap, z0, verbose, vos);
        if (!nr.converged) {
            if (verbose)
                os << "[info] laplaceLp: analytic L1 stalled; entering "
                      "eps-regularized smoothing.\n";
            nr = solveLnSmoothed(R, nlap, 1.0, z0, mm.errmax, verbose, vos);
        }
    } else {
        // ---- odd integer normP>=3 or non-integer normP>1: |eta|^p, split rule ----
        // Try the direct minimax warm start; if it stalls (low non-integer normP with
        // a singular |eta|^{p-2} Hessian), fall back to eps-regularized smoothing.
        std::ostream* vos = verbose ? &os : nullptr;
        // Direct probe with a tighter budget: well-behaved normP converge in a couple
        // of outer iterations, so a stall bails quickly into the smoothed fallback.
        nr = solveLnAbs(R, nlap, normP, z0, 6, 150, verbose, vos);
        if (!nr.converged) {
            if (verbose)
                os << "[info] laplaceLp: direct solve stalled at normP=" << normP
                   << "; entering eps-regularized smoothing.\n";
            nr = solveLnSmoothed(R, nlap, normP, z0, mm.errmax, verbose, vos);
        }
    }

    // 6. Back-transform to [ymin, ymax]; errmax = L_n norm on normalised [1, R].
    MinimaxResult out;
    out.expon.resize(nlap); out.weight.resize(nlap);
    for (int k = 0; k < nlap; ++k) {
        out.expon[k]  = DD::ddExp(nr.x[k]).hi / ymin;
        out.weight[k] = nr.x[nlap + k].hi / ymin;
    }
    // errmax = L_p norm on the normalised [1, R] domain (no ymin scaling):
    //   (∫_1^R |eta|^p dx)^{1/p},  with nr.fval = ∫_1^R |eta|^p dx the x-space
    // integral (incl. the e^t Jacobian) actually optimised.
    DD norm = DD::ddExp(DD::ddLog(nr.fval) / normP); // (L_p)^{1/p}
    out.errmax = norm.hi;
    if (verbose >= 3) detail::printCitationBlock(os);
    return out;
}

} // namespace minimax_cpppy
