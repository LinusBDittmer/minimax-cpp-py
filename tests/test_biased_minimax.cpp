// tests/test_biased_minimax.cpp
#include "test_helpers.hpp"
#include "core/algorithm_biased.hpp"
#include "molecular_orbital_data.hpp"
#include <chrono>
#include <cmath>
#include <vector>

using minimax_cpppy::detail::BiasedMgfContext;
using minimax_cpppy::detail::buildBiasedMgfContext;
using minimax_cpppy::detail::evalMgf;

// Ground-truth (unstabilised, direct summation) MGF for a small system where
// overflow is not a concern -- checks the shift-stabilised closed form against
// the textbook formula MGF(a) = e^{sigma^2 a^2} * [M_virt(a)*M_occ(a)]^n_exc / N^n_exc.
static void directMgf(const double* occ, int n_occ, const double* virt, int n_virt,
                      double bandwidth, int n_exc, double alpha,
                      double& mgf) {
    double D_single_max = virt[n_virt - 1] - occ[0];  // occ/virt must be pre-sorted ascending
    double sigma = bandwidth * (n_exc * D_single_max) / 4.0;
    double M_virt = 0.0, M_occ = 0.0;
    for (int a = 0; a < n_virt; ++a) M_virt += std::exp(alpha * virt[a]);
    for (int i = 0; i < n_occ;  ++i) M_occ  += std::exp(-alpha * occ[i]);
    double N = static_cast<double>(n_occ) * static_cast<double>(n_virt);
    double S = M_virt * M_occ;
    mgf = std::exp(sigma * sigma * alpha * alpha) * std::pow(S, n_exc) / std::pow(N, n_exc);
}

MINIMAX_TEST(mgf_matches_direct_summation) {
    double occ[]  = {-1.0, -0.6};
    double virt[] = {0.3, 0.9, 1.5};
    double bandwidth = 1.0;
    int n_exc = 1;
    auto ctx = buildBiasedMgfContext(occ, 2, virt, 3, bandwidth, n_exc, /*ymin=*/1.0);

    for (double alpha : {0.1, 0.5, 1.3}) {
        double mgf, dmgf, d2mgf, mgf_direct;
        evalMgf(ctx, alpha, mgf, dmgf, d2mgf);
        directMgf(occ, 2, virt, 3, bandwidth, n_exc, alpha, mgf_direct);
        MINIMAX_REQUIRE_CLOSE(mgf, mgf_direct, 1e-10);
    }
}

MINIMAX_TEST(mgf_derivatives_match_finite_difference) {
    double occ[]  = {-1.0, -0.6};
    double virt[] = {0.3, 0.9, 1.5};
    auto ctx = buildBiasedMgfContext(occ, 2, virt, 3, 1.0, 2, /*ymin=*/1.0);

    const double alpha = 0.4;
    const double eps = 1e-6;
    double mgf, dmgf, d2mgf;
    evalMgf(ctx, alpha, mgf, dmgf, d2mgf);

    double mp, dp, d2p, mm, dm, d2m;
    evalMgf(ctx, alpha + eps, mp, dp, d2p);
    evalMgf(ctx, alpha - eps, mm, dm, d2m);

    double fd1 = (mp - mm) / (2 * eps);
    double fd2 = (dp - dm) / (2 * eps);
    MINIMAX_REQUIRE_CLOSE(dmgf, fd1, 1e-6);
    MINIMAX_REQUIRE_CLOSE(d2mgf, fd2, 1e-5);
}

// Guards the "no O(n_occ * n_virt) pairwise loop" invariant: evalMgf's per-call
// cost must grow linearly with orbital count, not quadratically. Ideal ratio for
// a 10x orbital-count increase is ~10x; a generous 30x bound tolerates timing
// noise while still catching an O(n^2) regression (which would show ~100x).
MINIMAX_TEST(mgf_scales_linearly_in_orbitals) {
    auto timeEvalMgf = [](int n) -> double {
        std::vector<double> occ(n), virt(n);
        for (int i = 0; i < n; ++i) { occ[i] = -1.0 - 0.001 * i; virt[i] = 0.5 + 0.001 * i; }
        auto ctx = buildBiasedMgfContext(occ.data(), n, virt.data(), n, 1.0, 2, 1.0);
        double mgf, dmgf, d2mgf;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 200; ++rep) {
            double alpha = 0.01 + 0.0001 * rep;
            evalMgf(ctx, alpha, mgf, dmgf, d2mgf);
        }
        auto t1 = std::chrono::steady_clock::now();
        (void)mgf; (void)dmgf; (void)d2mgf;
        return std::chrono::duration<double>(t1 - t0).count();
    };
    double tSmall = timeEvalMgf(200);
    double tLarge = timeEvalMgf(2000);
    MINIMAX_REQUIRE(tLarge / std::max(tSmall, 1e-9) < 30.0);
}

using minimax_cpppy::detail::DD;
using minimax_cpppy::detail::evalBiasedMgfResidual;

// The additive constant c0 does not affect gradient/Hessian, so an arbitrary
// fixed value is fine for this test -- only the (a,w)-dependence is checked.
MINIMAX_TEST(mgf_residual_gradient_matches_finite_difference) {
    double occ[]  = {-1.0, -0.6, -0.4};
    double virt[] = {0.3, 0.9, 1.5, 2.2};
    const double ymin = 1.7;  // non-trivial, exercises the 1/ymin chain-rule factor
    auto ctx = buildBiasedMgfContext(occ, 3, virt, 4, 1.0, 2, ymin);
    const DD c0(3.7);

    const int nlap = 2;
    std::vector<DD> a = {DD(0.3), DD(1.2)};
    std::vector<DD> w = {DD(0.5), DD(0.8)};
    std::vector<DD> gA(nlap), gW(nlap);
    evalBiasedMgfResidual(ctx, c0, nlap, a.data(), w.data(), gA.data(), gW.data(), nullptr);

    const double eps = 1e-6;
    for (int k = 0; k < nlap; ++k) {
        std::vector<DD> ap = a, am = a;
        ap[k] = DD(a[k].hi + eps); am[k] = DD(a[k].hi - eps);
        DD rp = evalBiasedMgfResidual(ctx, c0, nlap, ap.data(), w.data(), nullptr, nullptr, nullptr);
        DD rm = evalBiasedMgfResidual(ctx, c0, nlap, am.data(), w.data(), nullptr, nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gA[k].hi, (rp.hi - rm.hi) / (2 * eps), 1e-6);

        std::vector<DD> wp = w, wm = w;
        wp[k] = DD(w[k].hi + eps); wm[k] = DD(w[k].hi - eps);
        DD rwp = evalBiasedMgfResidual(ctx, c0, nlap, a.data(), wp.data(), nullptr, nullptr, nullptr);
        DD rwm = evalBiasedMgfResidual(ctx, c0, nlap, a.data(), wm.data(), nullptr, nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gW[k].hi, (rwp.hi - rwm.hi) / (2 * eps), 1e-6);
    }
}

MINIMAX_TEST(mgf_residual_hessian_matches_finite_difference) {
    double occ[]  = {-1.0, -0.6, -0.4};
    double virt[] = {0.3, 0.9, 1.5, 2.2};
    const double ymin = 1.7;
    auto ctx = buildBiasedMgfContext(occ, 3, virt, 4, 1.0, 2, ymin);
    const DD c0(3.7);

    const int nlap = 2;
    const int dim = 2 * nlap;
    std::vector<DD> a = {DD(0.3), DD(1.2)};
    std::vector<DD> w = {DD(0.5), DD(0.8)};
    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    evalBiasedMgfResidual(ctx, c0, nlap, a.data(), w.data(), gA.data(), gW.data(), H.data());

    const double eps = 1e-5;
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + eps);
                        wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        std::vector<DD> gpA(nlap), gpW(nlap), gmA(nlap), gmW(nlap);
        evalBiasedMgfResidual(ctx, c0, nlap, ap.data(), wp.data(), gpA.data(), gpW.data(), nullptr);
        evalBiasedMgfResidual(ctx, c0, nlap, am.data(), wm.data(), gmA.data(), gmW.data(), nullptr);
        for (int i = 0; i < dim; ++i) {
            double gp = (i < nlap) ? gpA[i].hi : gpW[i - nlap].hi;
            double gm = (i < nlap) ? gmA[i].hi : gmW[i - nlap].hi;
            MINIMAX_REQUIRE_CLOSE(H[i + j * dim].hi, (gp - gm) / (2 * eps), 5e-5);
        }
    }
}

using minimax_cpppy::detail::squareResidual;

// Synthetic scalar r(theta) = a0^2 + 2*a0*w0 - w1 (theta = [a0,a1,w0,w1], nlap=2)
// with a hand-derived gradient/Hessian, used only to validate squareResidual's
// J=r^2 chain rule in isolation from the MGF/quadrature residuals.
static DD syntheticR(const DD* a, const DD* w, DD* gA, DD* gW, DD* hAW) {
    DD r = DD::ddSquare(a[0]) + DD(2.0) * a[0] * w[0] - w[1];
    const int dim = 4;
    if (gA) { gA[0] = DD(2.0) * a[0] + DD(2.0) * w[0]; gA[1] = DD(0.0); }
    if (gW) { gW[0] = DD(2.0) * a[0]; gW[1] = DD(-1.0); }
    if (hAW) {
        for (int i = 0; i < dim * dim; ++i) hAW[i] = DD(0.0);
        hAW[0 + 0 * dim] = DD(2.0);             // d2r/da0^2
        hAW[0 + 2 * dim] = DD(2.0);             // d2r/da0 dw0
        hAW[2 + 0 * dim] = DD(2.0);
    }
    return r;
}

MINIMAX_TEST(square_residual_matches_finite_difference) {
    const int nlap = 2, dim = 4;
    std::vector<DD> a = {DD(0.7), DD(1.3)}, w = {DD(0.4), DD(0.9)};
    std::vector<DD> gA(nlap), gW(nlap), hAW(dim * dim);
    DD r = syntheticR(a.data(), w.data(), gA.data(), gW.data(), hAW.data());

    std::vector<DD> gAJ(nlap), gWJ(nlap), hAWJ(dim * dim);
    squareResidual(nlap, r, gA.data(), gW.data(), hAW.data(), gAJ.data(), gWJ.data(), hAWJ.data());

    const double eps = 1e-5;
    auto evalJ = [&](const DD* ap, const DD* wp) -> double {
        DD rp = syntheticR(ap, wp, nullptr, nullptr, nullptr);
        return (rp * rp).hi;
    };
    // Gradient check
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + eps);
                        wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        double fd = (evalJ(ap.data(), wp.data()) - evalJ(am.data(), wm.data())) / (2 * eps);
        double an = (j < nlap) ? gAJ[j].hi : gWJ[j - nlap].hi;
        MINIMAX_REQUIRE_CLOSE(an, fd, 1e-6);
    }
    // Hessian check (FD of the analytic gradient via syntheticR + squareResidual again)
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + eps);
                        wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        std::vector<DD> gpA(nlap), gpW(nlap), gmA(nlap), gmW(nlap);
        DD rp = syntheticR(ap.data(), wp.data(), gpA.data(), gpW.data(), nullptr);
        DD rm = syntheticR(am.data(), wm.data(), gmA.data(), gmW.data(), nullptr);
        std::vector<DD> gpAJ(nlap), gpWJ(nlap), gmAJ(nlap), gmWJ(nlap);
        squareResidual(nlap, rp, gpA.data(), gpW.data(), nullptr, gpAJ.data(), gpWJ.data(), nullptr);
        squareResidual(nlap, rm, gmA.data(), gmW.data(), nullptr, gmAJ.data(), gmWJ.data(), nullptr);
        for (int i = 0; i < dim; ++i) {
            double gp = (i < nlap) ? gpAJ[i].hi : gpWJ[i - nlap].hi;
            double gm = (i < nlap) ? gmAJ[i].hi : gmWJ[i - nlap].hi;
            MINIMAX_REQUIRE_CLOSE(hAWJ[i + j * dim].hi, (gp - gm) / (2 * eps), 1e-5);
        }
    }
}

using minimax_cpppy::detail::evalBiasedConstant;
using minimax_cpppy::detail::compositeGaussLegendre;
using minimax_cpppy::DenominatorDensity;

MINIMAX_TEST(biased_constant_panel_doubling_converges) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);

    DD c_64  = evalBiasedConstant(d, 64);
    DD c_128 = evalBiasedConstant(d, 128);
    DD c_256 = evalBiasedConstant(d, 256);

    double d1 = std::abs(c_128.hi - c_64.hi);
    double d2 = std::abs(c_256.hi - c_128.hi);
    MINIMAX_REQUIRE(d2 < d1 * 0.5 + 1e-14);  // converging, not oscillating/diverging
    MINIMAX_REQUIRE(std::isfinite(c_256.hi) && c_256.hi > 0.0);
}

MINIMAX_TEST(biased_constant_matches_direct_integration) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);
    DD c0 = evalBiasedConstant(d, 512);

    // Independent check: integrate e^{-t} rho(t) dt directly against evalW on a
    // fine uniform grid (trapezoidal), not the composite-GL machinery under test.
    double b = std::log(d.ratio());
    const int N = 200000;
    double integral = 0.0;
    double prevVal = 0.0;
    for (int j = 0; j <= N; ++j) {
        double t = b * j / N;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        double val = std::exp(-t) * w;
        if (j > 0) integral += 0.5 * (val + prevVal) * (b / N);
        prevVal = val;
    }
    MINIMAX_REQUIRE_CLOSE(c0.hi, integral, 1e-6);
}

using minimax_cpppy::detail::evalBiasedLoss;
using minimax_cpppy::detail::QuadRule;

// Value check: evalBiasedLoss against an independent trapezoidal integration
// of eta(x)*rho(t) dt on the same domain (not the composite-GL rule under test).
MINIMAX_TEST(biased_loss_matches_independent_integration) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    DenominatorDensity density(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);
    const int nlap = 2;
    std::vector<DD> a = {DD(0.3), DD(1.5)}, w = {DD(0.4), DD(0.7)};

    double b = std::log(density.ratio());
    QuadRule rule = compositeGaussLegendre(0.0, b, 512);
    DD r = evalBiasedLoss(rule, nlap, a.data(), w.data(), density, nullptr, nullptr, nullptr);

    const int N = 200000;
    double integral = 0.0, prevVal = 0.0;
    for (int j = 0; j <= N; ++j) {
        double t = b * j / N;
        double x = std::exp(t);
        double eta = std::exp(-t);
        for (int k = 0; k < nlap; ++k) eta -= w[k].hi * std::exp(-a[k].hi * x);
        double rho, drho, d2rho;
        density.evalW(t, rho, drho, d2rho);
        double val = eta * rho;
        if (j > 0) integral += 0.5 * (val + prevVal) * (b / N);
        prevVal = val;
    }
    MINIMAX_REQUIRE_CLOSE(r.hi, integral, 1e-6);
}

MINIMAX_TEST(biased_loss_gradient_matches_finite_difference) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    DenominatorDensity density(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);
    const int nlap = 2;
    double b = std::log(density.ratio());
    QuadRule rule = compositeGaussLegendre(0.0, b, 512);

    std::vector<DD> a = {DD(0.3), DD(1.5)}, w = {DD(0.4), DD(0.7)};
    std::vector<DD> gA(nlap), gW(nlap);
    evalBiasedLoss(rule, nlap, a.data(), w.data(), density, gA.data(), gW.data(), nullptr);

    const double eps = 1e-6;
    for (int k = 0; k < nlap; ++k) {
        std::vector<DD> ap = a, am = a;
        ap[k] = DD(a[k].hi + eps); am[k] = DD(a[k].hi - eps);
        DD rp = evalBiasedLoss(rule, nlap, ap.data(), w.data(), density, nullptr, nullptr, nullptr);
        DD rm = evalBiasedLoss(rule, nlap, am.data(), w.data(), density, nullptr, nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gA[k].hi, (rp.hi - rm.hi) / (2 * eps), 1e-6);

        std::vector<DD> wp = w, wm = w;
        wp[k] = DD(w[k].hi + eps); wm[k] = DD(w[k].hi - eps);
        DD rwp = evalBiasedLoss(rule, nlap, a.data(), wp.data(), density, nullptr, nullptr, nullptr);
        DD rwm = evalBiasedLoss(rule, nlap, a.data(), wm.data(), density, nullptr, nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gW[k].hi, (rwp.hi - rwm.hi) / (2 * eps), 1e-6);
    }
}

MINIMAX_TEST(biased_loss_hessian_matches_finite_difference) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    DenominatorDensity density(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);
    const int nlap = 2, dim = 4;
    double b = std::log(density.ratio());
    QuadRule rule = compositeGaussLegendre(0.0, b, 512);

    std::vector<DD> a = {DD(0.3), DD(1.5)}, w = {DD(0.4), DD(0.7)};
    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    evalBiasedLoss(rule, nlap, a.data(), w.data(), density, gA.data(), gW.data(), H.data());

    const double eps = 1e-5;
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + eps);
                        wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        std::vector<DD> gpA(nlap), gpW(nlap), gmA(nlap), gmW(nlap);
        evalBiasedLoss(rule, nlap, ap.data(), wp.data(), density, gpA.data(), gpW.data(), nullptr);
        evalBiasedLoss(rule, nlap, am.data(), wm.data(), density, gmA.data(), gmW.data(), nullptr);
        for (int i = 0; i < dim; ++i) {
            double gp = (i < nlap) ? gpA[i].hi : gpW[i - nlap].hi;
            double gm = (i < nlap) ? gmA[i].hi : gmW[i - nlap].hi;
            MINIMAX_REQUIRE_CLOSE(H[i + j * dim].hi, (gp - gm) / (2 * eps), 5e-5);
        }
    }
}

using minimax_cpppy::detail::biasedLaplace;
using minimax_cpppy::MinimaxResult;

// Computes the literal r(theta) = int_0^b eta(t;theta) rho(t) dt for a converged
// result, using the SAME normalised-domain convention biasedLaplace
// works in internally (a_norm = expon*ymin, w_norm = weight*ymin).
static double literalBiasResidual(const MinimaxResult& r, double ymin, double ymax,
                                  const DenominatorDensity& density, int panels) {
    double b = std::log(ymax / ymin);
    QuadRule rule = compositeGaussLegendre(0.0, b, panels);
    double bias = 0.0;
    for (size_t i = 0; i < rule.t.size(); ++i) {
        double t = rule.t[i], om = rule.w[i];
        double x = std::exp(t);
        double eta = 1.0 / x;
        for (size_t k = 0; k < r.expon.size(); ++k)
            eta -= (r.weight[k] * ymin) * std::exp(-(r.expon[k] * ymin) * x);
        double rho, drho, d2rho;
        density.evalW(t, rho, drho, d2rho);
        bias += om * eta * rho;
    }
    return bias;
}

MINIMAX_TEST(biased_reduces_bias_vs_unbiased_warm_start) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    DenominatorDensity density(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);
    double ymin = density.deltaMin(), ymax = density.deltaMax();
    int nlap = 4;

    auto r_unbiased = minimax_cpppy::laplaceMinimax(nlap, ymin, ymax);
    double bias_before = std::abs(literalBiasResidual(r_unbiased, ymin, ymax, density, 512));

    MinimaxResult r_biased = minimax_cpppy::biasedLaplace(
        nlap, ymin, ymax, m.occ, m.n_occ, m.virt, m.n_virt, 1.0);
    double bias_after = std::abs(literalBiasResidual(r_biased, ymin, ymax, density, 512));

    MINIMAX_REQUIRE(std::isfinite(r_biased.errmax) && r_biased.errmax > 0.0);
    MINIMAX_REQUIRE(static_cast<int>(r_biased.expon.size()) == nlap);
    for (int k = 0; k < nlap; ++k) {
        MINIMAX_REQUIRE(r_biased.expon[k] > 0.0);
        MINIMAX_REQUIRE(r_biased.weight[k] > 0.0);
    }
    MINIMAX_REQUIRE(bias_after < bias_before * 0.1);
    // Correction, not a redesign: errmax should stay within an order of magnitude.
    MINIMAX_REQUIRE(r_biased.errmax < r_unbiased.errmax * 10.0);
}

MINIMAX_TEST(biased_ratio_mismatch_throws) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    DenominatorDensity density(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);
    MINIMAX_REQUIRE_THROW_TYPE(
        minimax_cpppy::biasedLaplace(3, density.deltaMin() * 2.0, density.deltaMax(),
                                              m.occ, m.n_occ, m.virt, m.n_virt, 1.0),
        std::invalid_argument);
}

MINIMAX_TEST(biased_public_api_default_args_match_explicit) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    DenominatorDensity density(m.occ, m.n_occ, m.virt, m.n_virt, 1.0, 4096, 512, 1e-3);
    double ymin = density.deltaMin(), ymax = density.deltaMax();
    auto r1 = minimax_cpppy::biasedLaplace(3, ymin, ymax, m.occ, m.n_occ, m.virt, m.n_virt, 1.0);
    auto r2 = minimax_cpppy::biasedLaplace(
        3, ymin, ymax, m.occ, m.n_occ, m.virt, m.n_virt,
        1.0, 4096, 512, 1e-3, -1.0, 0.0, 2, /*verbose=*/0);
    for (int k = 0; k < 3; ++k) {
        MINIMAX_REQUIRE(r1.expon[k]  == r2.expon[k]);
        MINIMAX_REQUIRE(r1.weight[k] == r2.weight[k]);
    }
}

// Near-degenerate: occ[max] ~= virt[min], so physicalLogRatio ~= 0 < 0.01 and
// buildDensityArrays (called internally to build the density) clamps
// density.ratio() to exp(0.01). biasedLaplace must accept
// (deltaMin, deltaMax) computed the same way without throwing the ratio-
// mismatch error.
MINIMAX_TEST(biased_degenerate_density_does_not_throw) {
    double occ[]  = {1.0 - 0.001};
    double virt[] = {1.0 + 0.001};
    DenominatorDensity d_tmp(occ, 1, virt, 1, 0.1);
    MINIMAX_REQUIRE(d_tmp.ratio() > 1.0);  // clamped to exp(0.01) > 1
    double ymin = d_tmp.deltaMin(), ymax = d_tmp.deltaMax();
    try {
        (void)minimax_cpppy::biasedLaplace(1, ymin, ymax, occ, 1, virt, 1, 0.1);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(
            std::string("biased_degenerate_density_does_not_throw: "
                        "got unexpected invalid_argument (ratio mismatch?): ") + e.what());
    } catch (...) {
        // Any other exception (e.g. convergence) is acceptable on this near-trivial interval.
    }
}

MINIMAX_TEST(biased_stays_sane_on_known_hard_phase1_seed) {
    // Truncated (6-decimal) Ne fixture from tests/python/test_biasing.py --
    // originally confirmed to drive Phase-2 Newton into a diverged state
    // (expon[0]->0, weight[0]->-1.17e34) before the divergence guard was
    // added, and later (once Phase-2's own objective started arbitrating
    // between the Phase-1 seed and the raw unbiased warm start) confirmed to
    // converge cleanly instead of needing the guard at all -- this fixture no
    // longer exercises the fallback path, only the "stays sane" invariant.
    // The guard itself (isfinite/positivity/errmax-bound checks in
    // biasedLaplace) has no dedicated direct test; it is exercised
    // indirectly whenever a real input needs it.
    double occ[]  = {-32.765635, -1.918798, -0.832097, -0.832097, -0.832097};
    double virt[] = {1.694558, 1.694558, 1.694558, 2.159425, 5.196711,
                      5.196711, 5.196711, 5.196711, 5.196711};
    int n_occ = 5, n_virt = 9, nlap = 4;

    DenominatorDensity density(occ, n_occ, virt, n_virt, 1.0, 4096, 512, 1e-3);
    double ymin = density.deltaMin(), ymax = density.deltaMax();

    auto r_unbiased = minimax_cpppy::laplaceMinimax(nlap, ymin, ymax);
    auto r_biased = minimax_cpppy::biasedLaplace(
        nlap, ymin, ymax, occ, n_occ, virt, n_virt, 1.0);

    MINIMAX_REQUIRE(std::isfinite(r_biased.errmax));
    MINIMAX_REQUIRE(r_biased.errmax < r_unbiased.errmax * 10.0);
    for (int k = 0; k < nlap; ++k) {
        MINIMAX_REQUIRE(std::isfinite(r_biased.expon[k]) && r_biased.expon[k] > 0.0);
        MINIMAX_REQUIRE(std::isfinite(r_biased.weight[k]) && r_biased.weight[k] > 0.0);
    }
}

int main() { MINIMAX_RUN_TESTS(); }
