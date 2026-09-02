/**
 * @file algorithm_biased.hpp
 * @brief biased: density-uncorrelated bias-correction pathway.
 *
 * Drives the density-weighted net signed bias r(theta) = int_1^R eta(x) p(x) dx
 * to zero, starting from the unbiased minimax warm start. Two phases:
 *   Phase 1 -- closed-form Fourier/Laplace-domain MGF approximation (no
 *             quadrature, no FFT; O(nlap*(n_occ+n_virt)) per Newton iteration).
 *   Phase 2 -- exact t-space quadrature polish against the real, corrected
 *             DenominatorDensity, starting from Phase 1's near-root.
 * See docs/superpowers/specs/2026-07-09-biased-ud-loss-design.md for the full
 * derivation.
 */
#pragma once
#include "algorithm.hpp"
#include "dd128.hpp"
#include "newton.hpp"
#include "quadrature.hpp"
#include "minimax_cpppy/biasing.hpp"
#include <algorithm>
#include <cmath>
#include <ostream>
#include <vector>

namespace minimax_cpppy {
namespace detail {

/**
 * @brief Precomputed, orbital-derived context for the Phase-1 closed-form MGF.
 *
 * Computed once per biasedLaplace call and shared across every Newton
 * iteration -- no re-sort or re-scan of the orbital arrays per objective call.
 * delta[a] = eps_a,max - eps~_a >= 0 (virtual, shift-stabilised);
 * gamma[i] = eps~_i - eps_i,min >= 0 (occupied, shift-stabilised).
 */
struct BiasedMgfContext {
    std::vector<double> delta;  ///< eps_a,max - eps~_a, per virtual orbital (>= 0)
    std::vector<double> gamma;  ///< eps~_i - eps_i,min, per occupied orbital (>= 0)
    double D     = 0.0;         ///< eps_a,max - eps_i,min (= D_single_max)
    double sigma = 0.0;         ///< Gaussian kernel std dev (density.hpp's kernelStdDev)
    double N     = 0.0;         ///< n_occ * n_virt
    int    n_exc = 1;
    double ymin  = 1.0;         ///< Delta_min; alpha_k = a_k/ymin (physical exponent)
};

/**
 * @brief Build the Phase-1 MGF context from raw orbital energies.
 *
 * Mirrors buildDensityArrays' Step-1 sort/bounds computation (density.hpp) --
 * intentionally duplicated (a handful of lines) rather than shared, since the
 * two call sites need different outputs.
 */
inline BiasedMgfContext buildBiasedMgfContext(
    const double* occ, int n_occ, const double* virt, int n_virt,
    double bandwidth, int n_exc, double ymin)
{
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (n_occ < 1 || n_virt < 1) {
        throw std::invalid_argument("buildBiasedMgfContext: n_occ and n_virt must be >= 1");
    }
#endif
    std::vector<double> occSorted(occ, occ + n_occ);
    std::vector<double> virtSorted(virt, virt + n_virt);
    std::stable_sort(occSorted.begin(), occSorted.end());
    std::stable_sort(virtSorted.begin(), virtSorted.end());

    const double epsAMax = virtSorted.back();
    const double epsIMin = occSorted.front();
    const double D_single_max = epsAMax - epsIMin;
    const double denominatorMaxPhysical = n_exc * D_single_max;

    BiasedMgfContext ctx;
    ctx.delta.resize(n_virt);
    for (int a = 0; a < n_virt; ++a) ctx.delta[a] = epsAMax - virtSorted[a];
    ctx.gamma.resize(n_occ);
    for (int i = 0; i < n_occ; ++i) ctx.gamma[i] = occSorted[i] - epsIMin;
    ctx.D     = D_single_max;
    ctx.sigma = bandwidth * denominatorMaxPhysical / 4.0;
    ctx.N     = static_cast<double>(n_occ) * static_cast<double>(n_virt);
    ctx.n_exc = n_exc;
    ctx.ymin  = ymin;
    return ctx;
}

/**
 * @brief Evaluate MGF(alpha), MGF'(alpha), MGF''(alpha) in closed form.
 *
 * MGF(alpha) = E_raw[e^{-alpha*Delta}], the density's characteristic function
 * continued to imaginary frequency (real exponentials only, no FFT/complex
 * arithmetic). O(n_occ + n_virt); plain double precision is adequate since this
 * is the fast/approximate Phase-1 objective -- Phase 2 supplies the exact,
 * DD-precision polish.
 */
inline void evalMgf(const BiasedMgfContext& ctx, double alpha,
                     double& mgf, double& dmgf, double& d2mgf)
{
    double Q0 = 0.0, Q1 = 0.0, Q2 = 0.0;
    for (double d : ctx.delta) {
        double e = std::exp(-alpha * d);
        Q0 += e; Q1 += d * e; Q2 += d * d * e;
    }
    double P0 = 0.0, P1 = 0.0, P2 = 0.0;
    for (double g : ctx.gamma) {
        double e = std::exp(-alpha * g);
        P0 += e; P1 += g * e; P2 += g * g * e;
    }

    const double L   = ctx.sigma * ctx.sigma * alpha * alpha
                      + ctx.n_exc * (alpha * ctx.D + std::log(Q0) + std::log(P0))
                      - ctx.n_exc * std::log(ctx.N);
    const double Lp  = 2.0 * ctx.sigma * ctx.sigma * alpha
                      + ctx.n_exc * (ctx.D - Q1 / Q0 - P1 / P0);
    const double Lpp = 2.0 * ctx.sigma * ctx.sigma
                      + ctx.n_exc * ((Q2 * Q0 - Q1 * Q1) / (Q0 * Q0)
                                   + (P2 * P0 - P1 * P1) / (P0 * P0));

    mgf   = std::exp(L);
    dmgf  = mgf * Lp;
    d2mgf = mgf * (Lp * Lp + Lpp);
}

/**
 * @brief Phase-1 residual r1(theta) = c0 - sum_k w_k * MGF(a_k/ymin), plus its
 * gradient and Hessian in (a,w)-space.
 *
 * c0 (the density's exact E_p[1/x], see evalBiasedConstant) is a fixed
 * constant supplied by the caller -- it does not depend on (a,w).
 * gradA/gradW (length nlap) and hessAW (2*nlap square, column-major,
 * a_k -> k, w_k -> nlap+k) are filled when non-null, matching evalLnLossAbs's
 * contract.
 */
inline DD evalBiasedMgfResidual(
    const BiasedMgfContext& ctx, const DD& c0,
    int nlap, const DD* a, const DD* w,
    DD* gradA, DD* gradW, DD* hessAW)
{
    const int dim = 2 * nlap;
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0);
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0);

    DD r1 = c0;
    const bool needG = (gradA || gradW || hessAW);
    for (int k = 0; k < nlap; ++k) {
        const double alpha = a[k].hi / ctx.ymin;
        double mgf, dmgf, d2mgf;
        evalMgf(ctx, alpha, mgf, dmgf, d2mgf);
        r1 = r1 - w[k] * DD(mgf);

        if (needG) {
            if (gradW) gradW[k] = DD(-mgf);
            if (gradA) gradA[k] = DD(-1.0) * w[k] * DD(dmgf) / DD(ctx.ymin);
            if (hessAW) {
                hessAW[k + k * dim] = DD(-1.0) * w[k] * DD(d2mgf) / DD(ctx.ymin * ctx.ymin);
                DD haw = DD(-1.0) * DD(dmgf) / DD(ctx.ymin);
                hessAW[k + (nlap + k) * dim] = haw;
                hessAW[(nlap + k) + k * dim] = haw;
            }
        }
    }
    return r1;
}

/**
 * @brief Chain rule for J(theta) = r(theta)^2 given r's own (a,w)-space
 * gradient/Hessian: grad J = 2*r*grad(r), Hess J = 2*grad(r)*grad(r)^T +
 * 2*r*Hess(r). Shared by both Phase-1 (evalBiasedMgfResidual) and Phase-2
 * (evalBiasedLoss) objectives -- the squaring math is identical for both.
 * hAW_J may be nullptr (gradient-only; matches gA_J/gW_J then being sufficient).
 */
inline void squareResidual(
    int nlap, const DD& r,
    const DD* gA_r, const DD* gW_r, const DD* hAW_r,
    DD* gA_J, DD* gW_J, DD* hAW_J)
{
    const int dim = 2 * nlap;
    if (gA_J) for (int k = 0; k < nlap; ++k) gA_J[k] = DD(2.0) * r * gA_r[k];
    if (gW_J) for (int k = 0; k < nlap; ++k) gW_J[k] = DD(2.0) * r * gW_r[k];
    if (hAW_J) {
        auto g = [&](int i) -> DD { return i < nlap ? gA_r[i] : gW_r[i - nlap]; };
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                hAW_J[i + j * dim] = DD(2.0) * g(i) * g(j) + DD(2.0) * r * hAW_r[i + j * dim];
    }
}

/**
 * @brief One-shot constant c0 = E_p[1/x] = int_0^b e^{-t} rho(t) dt.
 *
 * Computed once per biasedLaplace call (not per Newton iteration),
 * against the real, corrected DenominatorDensity -- this is the quantity
 * Phase 1's raw MGF model cannot supply (E_raw[1/Delta] is ill-defined; see
 * the spec's Formulation section).
 */
inline DD evalBiasedConstant(const DenominatorDensity& density, int panels) {
    const double b = std::log(density.ratio());
    QuadRule rule = compositeGaussLegendre(0.0, b, panels);
    DD c0(0.0);
    for (size_t i = 0; i < rule.t.size(); ++i) {
        DD t(rule.t[i]);
        DD om(rule.w[i]);
        double rho, drho, d2rho;
        density.evalW(rule.t[i], rho, drho, d2rho);
        c0 = c0 + om * DD::ddExp(-t) * DD(rho);
    }
    return c0;
}

/**
 * @brief Phase-2 literal residual r(theta) = int_0^b eta(t;theta) rho(t) dt,
 * exact against the real DenominatorDensity, plus its gradient/Hessian.
 *
 * No boundary terms (unlike the L1 zero-splitting engine): the integration
 * limits [0,b] are fixed constants, so differentiating under the integral
 * sign simply replaces eta by its (a,w)-derivatives. Mirrors evalLnLossAbs's
 * per-node accumulation pattern with the extra rho(t) weight and no |.|^n
 * exponent (eta enters signed, linearly).
 */
inline DD evalBiasedLoss(
    const QuadRule& rule, int nlap,
    const DD* a, const DD* w,
    const DenominatorDensity& density,
    DD* gradA, DD* gradW, DD* hessAW)
{
    const int dim = 2 * nlap;
    DD r(0.0);
    if (gradA) for (int k = 0; k < nlap; ++k) gradA[k] = DD(0.0);
    if (gradW) for (int k = 0; k < nlap; ++k) gradW[k] = DD(0.0);
    if (hessAW) for (int i = 0; i < dim * dim; ++i) hessAW[i] = DD(0.0);

    const bool needG = (gradA || gradW || hessAW);
    std::vector<DD> bk(nlap);
    const size_t N = rule.t.size();
    for (size_t i = 0; i < N; ++i) {
        const DD t(rule.t[i]);
        const DD om(rule.w[i]);
        const DD x = DD::ddExp(t);
        DD eta = DD::ddExp(-t);
        for (int k = 0; k < nlap; ++k) {
            bk[k] = DD::ddExp(-(a[k] * x));
            eta = eta - w[k] * bk[k];
        }
        double rho, drho, d2rho;
        density.evalW(rule.t[i], rho, drho, d2rho);
        const DD omrho = om * DD(rho);
        r = r + omrho * eta;

        if (needG) {
            for (int k = 0; k < nlap; ++k) {
                const DD va = w[k] * (x * bk[k]);   // eta_{a_k}
                const DD vw = -bk[k];               // eta_{w_k}
                if (gradA) gradA[k] = gradA[k] + omrho * va;
                if (gradW) gradW[k] = gradW[k] + omrho * vw;
                if (hessAW) {
                    const DD x2bk = x * (x * bk[k]);
                    hessAW[k + k * dim] = hessAW[k + k * dim] - omrho * (w[k] * x2bk);
                    const DD haw = omrho * (x * bk[k]);
                    hessAW[k + (nlap + k) * dim] = hessAW[k + (nlap + k) * dim] + haw;
                    hessAW[(nlap + k) + k * dim] = hessAW[(nlap + k) + k * dim] + haw;
                }
            }
        }
    }
    return r;
}

// chain rule a_k = e^{p_k}: map (a,w)-space grad/Hess into z=(p,w)-space.
// Duplicated from src/laplace_lp.cpp (a different translation unit) rather than
// shared -- ~15 lines, not worth a new header for.
//   grad_p_k = a_k g_{a_k};  grad_w_k = g_{w_k}
//   H_{p_k p_l} = a_k a_l H_{a_k a_l} + delta_kl a_k g_{a_k}
//   H_{p_k w_l} = a_k H_{a_k w_l};   H_{w_k w_l} = H_{w_k w_l}
inline void chainRuleToZ(int nlap, int dim, const DD* a,
                         const DD* gA, const DD* gW,
                         const DD* hAW, DD* grad, DD* hess) {
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

/**
 * @brief biased entry point: two-phase Newton solve for the density-
 * uncorrelated bias correction, warm-started from the unbiased minimax.
 *
 * Phase 1 (closed-form, fast, approximate): Newton on J1=r1^2 using the raw
 * Fourier/Laplace-domain MGF -- no quadrature, no FFT.
 * Phase 2 (exact, from Phase 1's near-root): Newton on the literal J=r^2 via
 * composite-GL quadrature against the real, corrected DenominatorDensity.
 *
 * Requires the density implied by (occ, virt, bandwidth, n_fft, n_t,
 * floor_frac, floor_frac_max, C, n_exc) to have ratio() matching ymax/ymin
 * within 1 ppm (same check/degenerate-ratio exception as biasedLaplace).
 *
 * errmax in the returned MinimaxResult is the classic normalised-domain
 * max|eta_norm(x_norm)| (same convention as laplaceMinimax/biasedLaplace
 * -- NOT r(theta), which is a different, smaller quantity by design).
 */
inline minimax_cpppy::MinimaxResult biasedLaplace(
    int nlap, double ymin, double ymax,
    const double* occ, int n_occ, const double* virt, int n_virt,
    double bandwidth, int n_fft, int n_t,
    double floor_frac, double floor_frac_max, double C, int n_exc,
    int maxIter, double toleranceMaehly, double toleranceNR,
    double stepMax, double delta, double armijoConstant,
    int verbose, std::ostream* os)
{
    minimax_cpppy::DenominatorDensity density(
        occ, n_occ, virt, n_virt, bandwidth, n_fft, n_t,
        floor_frac, floor_frac_max, C, n_exc);

    const double ratio = ymax / ymin;
    const double densityRatio = density.ratio();
    const bool isDegenerate = (std::log(densityRatio) < 0.01 + 1e-9);
    if (!isDegenerate && std::abs(densityRatio - ratio) / ratio > 1e-6)
        throw std::invalid_argument(
            "biasedLaplace: density ratio (from occ/virt) does not match ymax/ymin");

    minimax_cpppy::MinimaxResult unbiasedResult = laplaceMinimax(
        nlap, ymin, ymax, maxIter, toleranceMaehly, toleranceNR,
        stepMax, delta, armijoConstant, verbose, os);

    const int dim = 2 * nlap;
    std::vector<DD> z0(dim);
    for (int k = 0; k < nlap; ++k) {
        z0[k]        = DD::ddLog(DD(unbiasedResult.expon[k] * ymin));
        z0[nlap + k] = DD(unbiasedResult.weight[k] * ymin);
    }

    BiasedMgfContext ctx = buildBiasedMgfContext(occ, n_occ, virt, n_virt, bandwidth, n_exc, ymin);
    const int panels = std::max(64, 8 * nlap);
    DD c0 = evalBiasedConstant(density, panels);

    std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap), gAJ(nlap), gWJ(nlap);
    std::vector<DD> hAW(static_cast<size_t>(dim) * dim), hAWJ(static_cast<size_t>(dim) * dim);

    auto fPhase1 = [&](const DD* z, DD* grad, DD* hess) -> DD {
        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(z[k]); w[k] = z[nlap + k]; }
        const bool needG = (grad || hess);
        DD r1 = evalBiasedMgfResidual(ctx, c0, nlap, a.data(), w.data(),
                                        needG ? gA.data() : nullptr,
                                        needG ? gW.data() : nullptr,
                                        hess ? hAW.data() : nullptr);
        if (grad || hess) {
            squareResidual(nlap, r1, gA.data(), gW.data(), hAW.data(),
                           gAJ.data(), gWJ.data(), hess ? hAWJ.data() : nullptr);
            chainRuleToZ(nlap, dim, a.data(), gAJ.data(), gWJ.data(),
                        hess ? hAWJ.data() : nullptr, grad, hess);
        }
        return r1 * r1;
    };
    NewtonResult nr1 = newtonMinimize(dim, fPhase1, z0.data(), maxIter, toleranceNR,
                                      verbose, os);

    QuadRule rule = compositeGaussLegendre(0.0, std::log(ratio), panels);
    auto fPhase2 = [&](const DD* z, DD* grad, DD* hess) -> DD {
        for (int k = 0; k < nlap; ++k) { a[k] = DD::ddExp(z[k]); w[k] = z[nlap + k]; }
        const bool needG = (grad || hess);
        DD r2 = evalBiasedLoss(rule, nlap, a.data(), w.data(), density,
                                 needG ? gA.data() : nullptr,
                                 needG ? gW.data() : nullptr,
                                 hess ? hAW.data() : nullptr);
        if (grad || hess) {
            squareResidual(nlap, r2, gA.data(), gW.data(), hAW.data(),
                           gAJ.data(), gWJ.data(), hess ? hAWJ.data() : nullptr);
            chainRuleToZ(nlap, dim, a.data(), gAJ.data(), gWJ.data(),
                        hess ? hAWJ.data() : nullptr, grad, hess);
        }
        return r2 * r2;
    };

    // Phase 1's closed-form Gaussian-kernel MGF grows like exp(sigma^2*alpha^2)
    // in the exponents alpha -- mathematically fine, but for an unbiased warm
    // start with any alpha_k = O(1/sigma), it makes fPhase1 (and its Newton
    // step) numerically extreme (f ~ 1e100+) right at z0. Whether that first
    // Armijo trial then accepts or rejects a step is decided by last-ulp
    // differences in libm (std::exp/std::log) that vary across platforms, so
    // nr1.x can occasionally wander to a much worse seed than z0 itself even
    // though Phase 1's own (badly-scaled) objective decreased monotonically.
    // Arbitrate with Phase 2's real objective, which is what actually matters,
    // instead of trusting Phase 1's trajectory unconditionally.
    DD f2AtZ0  = fPhase2(z0.data(), nullptr, nullptr);
    DD f2AtNr1 = fPhase2(nr1.x.data(), nullptr, nullptr);
    const DD* phase2Seed = (std::isfinite(f2AtNr1.hi) && f2AtNr1.hi < f2AtZ0.hi)
                          ? nr1.x.data() : z0.data();
    NewtonResult nr2 = newtonMinimize(dim, fPhase2, phase2Seed, maxIter, toleranceNR,
                                      verbose, os);

    minimax_cpppy::MinimaxResult result;
    result.expon.resize(nlap);
    result.weight.resize(nlap);
    std::vector<double> aNorm(nlap), wNorm(nlap);
    for (int k = 0; k < nlap; ++k) {
        aNorm[k] = DD::ddExp(nr2.x[k]).hi;
        wNorm[k] = nr2.x[nlap + k].hi;
        result.expon[k]  = aNorm[k] / ymin;
        result.weight[k] = wNorm[k] / ymin;
    }

    // errmax: unscaled normalised-domain max|eta_norm| -- same convention as
    // laplaceMinimax/biasedLaplace (see Global Constraints).
    const double tmax = std::log(ratio);
    const int nScan = 2000;
    double errmax = 0.0;
    for (int j = 0; j <= nScan; ++j) {
        double t = tmax * j / nScan;
        double x = std::exp(t);
        double e = 1.0 / x;
        for (int k = 0; k < nlap; ++k) e -= wNorm[k] * std::exp(-aNorm[k] * x);
        errmax = std::max(errmax, std::abs(e));
    }
    // Divergence guard: Phase-2 Newton can occasionally diverge into a
    // pathological regime (a_k -> 0, w_k -> +-huge) for near-degenerate or
    // finely-perturbed inputs -- a narrow instability basin inherent to
    // Newton's method on this objective, not a flaw in the objective's own
    // math (already independently verified in Tasks 1-5's reviews). This is
    // a *correction*, not a redesign (see the spec's Formulation section):
    // if the result is non-finite/non-physical, or an order of magnitude
    // worse than the unbiased warm start, fall back to the unbiased result
    // rather than return a diverged one.
    bool resultIsSane = std::isfinite(errmax);
    for (int k = 0; resultIsSane && k < nlap; ++k) {
        resultIsSane = std::isfinite(result.expon[k]) && std::isfinite(result.weight[k])
                     && result.expon[k] > 0.0 && result.weight[k] > 0.0;
    }
    if (!resultIsSane || errmax > 10.0 * unbiasedResult.errmax) {
        if (verbose >= 1 && os) {
            *os << "[Biased WARN] Phase-2 Newton diverged (errmax=" << errmax
                << " vs unbiased=" << unbiasedResult.errmax
                << ", finite=" << resultIsSane
                << "); falling back to unbiased result\n";
        }
        return unbiasedResult;
    }

    result.errmax = errmax;
    return result;
}

} // namespace detail
} // namespace minimax_cpppy
