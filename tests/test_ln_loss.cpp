#include "test_helpers.hpp"
#include "core/ln_loss.hpp"
#include "core/dd128.hpp"
#include "core/expint.hpp"
#include "core/algorithm.hpp"
#include <cmath>
#include <vector>

using minimax_cpppy::detail::DD;
using minimax_cpppy::detail::QuadRule;
using minimax_cpppy::detail::compositeGaussLegendre;
using minimax_cpppy::detail::evalLnLoss;
using minimax_cpppy::detail::evalLnLossAbs;
using minimax_cpppy::detail::buildLnRule;
using minimax_cpppy::detail::findEtaZeros;
using minimax_cpppy::detail::findEtaZerosWarm;
using minimax_cpppy::detail::buildSplitRule;
using minimax_cpppy::detail::evalL1LossSmoothed;
using minimax_cpppy::detail::ddExpInt1;
using minimax_cpppy::detail::ddExpInt1Series;
using minimax_cpppy::detail::ddExpInt1CF;
using minimax_cpppy::detail::evalL2LossAnalytic;
using minimax_cpppy::detail::evalL1LossAnalytic;
using minimax_cpppy::detail::etaAt;
using minimax_cpppy::detail::etaAndPrimeAt;

// eta(t) in plain double, for an independent reference integrator.
static double etaD(double t, int nlap, const double* a, const double* w) {
    double x = std::exp(t);
    double s = std::exp(-t);
    for (int k = 0; k < nlap; ++k) s -= w[k] * std::exp(-a[k] * x);
    return s;
}
// Independent composite-Simpson reference for ∫_0^{lnR} eta^n e^t dt (x-space measure).
static double simpsonLn(double R, int nlap, int n, const double* a, const double* w) {
    const int M = 200000;               // even
    const double h = std::log(R) / M;
    double acc = 0.0;
    for (int i = 0; i <= M; ++i) {
        double t = i * h;
        double f = std::pow(etaD(t, nlap, a, w), n) * std::exp(t);
        double c = (i == 0 || i == M) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        acc += c * f;
    }
    return acc * h / 3.0;
}

MINIMAX_TEST(ln_loss_matches_independent_integration) {
    const int nlap = 3, n = 4;
    const double R = 50.0;
    double a[3] = {0.5, 2.0, 8.0};
    double w[3] = {0.4, 0.6, 1.5};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 4096);
    DD L = evalLnLoss(rule, nlap, n, aDD.data(), wDD.data(), nullptr, nullptr);
    double ref = simpsonLn(R, nlap, n, a, w);
    MINIMAX_REQUIRE_CLOSE(L.hi, ref, 1e-6);
}

MINIMAX_TEST(ln_loss_gradient_matches_finite_difference) {
    const int nlap = 2, n = 4;
    const double R = 30.0;
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 2048);
    std::vector<DD> gA(nlap), gW(nlap);
    evalLnLoss(rule, nlap, n, a.data(), w.data(), gA.data(), gW.data());

    const double eps = 1e-6;
    for (int k = 0; k < nlap; ++k) {
        std::vector<DD> ap = a; ap[k] = DD(a[k].hi + eps);
        std::vector<DD> am = a; am[k] = DD(a[k].hi - eps);
        DD Lp = evalLnLoss(rule, nlap, n, ap.data(), w.data(), nullptr, nullptr);
        DD Lm = evalLnLoss(rule, nlap, n, am.data(), w.data(), nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gA[k].hi, (Lp.hi - Lm.hi) / (2 * eps), 1e-5);

        std::vector<DD> wp = w; wp[k] = DD(w[k].hi + eps);
        std::vector<DD> wm = w; wm[k] = DD(w[k].hi - eps);
        DD Lwp = evalLnLoss(rule, nlap, n, a.data(), wp.data(), nullptr, nullptr);
        DD Lwm = evalLnLoss(rule, nlap, n, a.data(), wm.data(), nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gW[k].hi, (Lwp.hi - Lwm.hi) / (2 * eps), 1e-5);
    }
}

// buildLnRule returns a valid, well-resolved rule on [0, lnR].
MINIMAX_TEST(build_ln_rule_resolves_integral) {
    const int nlap = 3, n = 4;
    const double R = 50.0;
    double a[3] = {0.5, 2.0, 8.0};
    double w[3] = {0.4, 0.6, 1.5};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }

    QuadRule rule = buildLnRule(R, nlap, n, aDD.data(), wDD.data());

    // Weights sum to the interval length [0, lnR].
    double sw = 0.0;
    for (double ww : rule.w) sw += ww;
    MINIMAX_REQUIRE_CLOSE(sw, std::log(R), 1e-12);

    // Panels doubled at least once past the floor max(64, 8*nlap) -> >= 2*64 panels.
    MINIMAX_REQUIRE(static_cast<int>(rule.t.size()) >= 16 * 2 * 64);

    // The adaptive rule integrates eta^n to the independent reference.
    DD L = evalLnLoss(rule, nlap, n, aDD.data(), wDD.data(), nullptr, nullptr);
    double ref = simpsonLn(R, nlap, n, a, w);
    MINIMAX_REQUIRE_CLOSE(L.hi, ref, 1e-6);
}

// Concatenated (a,w)-space gradient into out[0..2K-1] (a_k -> k, w_k -> nlap+k).
static void gradVec(const QuadRule& rule, int nlap, int n,
                    const DD* a, const DD* w, DD* out) {
    std::vector<DD> gA(nlap), gW(nlap);
    evalLnLoss(rule, nlap, n, a, w, gA.data(), gW.data());
    for (int k = 0; k < nlap; ++k) { out[k] = gA[k]; out[nlap + k] = gW[k]; }
}

// Analytic (a,w)-space Hessian vs central finite difference of the gradient.
MINIMAX_TEST(ln_loss_hessian_matches_finite_difference) {
    const int nlap = 2, n = 4;
    const double R = 30.0;
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 2048);

    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    evalLnLoss(rule, nlap, n, a.data(), w.data(), gA.data(), gW.data(), H.data());

    const double eps = 1e-5;
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + eps);
                        wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        std::vector<DD> gp(dim), gm(dim);
        gradVec(rule, nlap, n, ap.data(), wp.data(), gp.data());
        gradVec(rule, nlap, n, am.data(), wm.data(), gm.data());
        for (int i = 0; i < dim; ++i) {
            double fd = (gp[i].hi - gm[i].hi) / (2 * eps);
            MINIMAX_REQUIRE_CLOSE(H[i + j * dim].hi, fd, 5e-5);
        }
    }
}

// Independent composite-Simpson reference for ∫_0^{lnR} |eta|^n e^t dt (x-space measure).
static double simpsonLnAbs(double R, int nlap, int n, const double* a, const double* w) {
    const int M = 200000;               // even
    const double h = std::log(R) / M;
    double acc = 0.0;
    for (int i = 0; i <= M; ++i) {
        double t = i * h;
        double f = std::pow(std::fabs(etaD(t, nlap, a, w)), n) * std::exp(t);
        double c = (i == 0 || i == M) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        acc += c * f;
    }
    return acc * h / 3.0;
}

// |eta|^n value matches independent integration for odd n (incl n=1).
MINIMAX_TEST(ln_loss_abs_matches_independent_integration) {
    const int nlap = 3;
    const double R = 50.0;
    double a[3] = {0.5, 2.0, 8.0};
    double w[3] = {0.4, 0.6, 1.5};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    for (int n : {1, 3, 5}) {
        DD L = evalLnLossAbs(rule, nlap, n, aDD.data(), wDD.data(), nullptr, nullptr);
        double ref = simpsonLnAbs(R, nlap, n, a, w);
        MINIMAX_REQUIRE_CLOSE(L.hi, ref, 1e-5);
    }
}

// Even n: evalLnLossAbs must equal evalLnLoss exactly (|eta|^n == eta^n).
MINIMAX_TEST(ln_loss_abs_equals_plain_for_even_n) {
    const int nlap = 3, n = 4;
    const double R = 50.0;
    double a[3] = {0.5, 2.0, 8.0};
    double w[3] = {0.4, 0.6, 1.5};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 1024);
    DD Lp = evalLnLoss   (rule, nlap, n, aDD.data(), wDD.data(), nullptr, nullptr);
    DD La = evalLnLossAbs(rule, nlap, n, aDD.data(), wDD.data(), nullptr, nullptr);
    MINIMAX_REQUIRE_CLOSE(La.hi, Lp.hi, 1e-14);
}

// Gradient of |eta|^n vs central finite difference, odd n and n=1.
MINIMAX_TEST(ln_loss_abs_gradient_matches_finite_difference) {
    const int nlap = 2;
    const double R = 30.0;
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    for (int n : {1, 3, 5}) {
        std::vector<DD> gA(nlap), gW(nlap);
        evalLnLossAbs(rule, nlap, n, a.data(), w.data(), gA.data(), gW.data());
        const double eps = 1e-6;
        for (int k = 0; k < nlap; ++k) {
            std::vector<DD> ap = a; ap[k] = DD(a[k].hi + eps);
            std::vector<DD> am = a; am[k] = DD(a[k].hi - eps);
            DD Lp = evalLnLossAbs(rule, nlap, n, ap.data(), w.data(), nullptr, nullptr);
            DD Lm = evalLnLossAbs(rule, nlap, n, am.data(), w.data(), nullptr, nullptr);
            MINIMAX_REQUIRE_CLOSE(gA[k].hi, (Lp.hi - Lm.hi) / (2 * eps), 1e-4);
            std::vector<DD> wp = w; wp[k] = DD(w[k].hi + eps);
            std::vector<DD> wm = w; wm[k] = DD(w[k].hi - eps);
            DD Lwp = evalLnLossAbs(rule, nlap, n, a.data(), wp.data(), nullptr, nullptr);
            DD Lwm = evalLnLossAbs(rule, nlap, n, a.data(), wm.data(), nullptr, nullptr);
            MINIMAX_REQUIRE_CLOSE(gW[k].hi, (Lwp.hi - Lwm.hi) / (2 * eps), 1e-4);
        }
    }
}

// Exact Hessian of |eta|^n vs central FD of the gradient, odd n>=3.
MINIMAX_TEST(ln_loss_abs_hessian_matches_finite_difference) {
    const int nlap = 2, n = 3;
    const double R = 30.0;
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    evalLnLossAbs(rule, nlap, n, a.data(), w.data(), gA.data(), gW.data(), H.data());
    const double eps = 1e-5;
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + eps);
                        wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        std::vector<DD> gpA(nlap), gpW(nlap), gmA(nlap), gmW(nlap);
        evalLnLossAbs(rule, nlap, n, ap.data(), wp.data(), gpA.data(), gpW.data());
        evalLnLossAbs(rule, nlap, n, am.data(), wm.data(), gmA.data(), gmW.data());
        for (int i = 0; i < dim; ++i) {
            double gp = (i < nlap) ? gpA[i].hi : gpW[i - nlap].hi;
            double gm = (i < nlap) ? gmA[i].hi : gmW[i - nlap].hi;
            MINIMAX_REQUIRE_CLOSE(H[i + j * dim].hi, (gp - gm) / (2 * eps), 5e-4);
        }
    }
}

// Zero-finder: located points satisfy eta(z) ~ 0 and lie ascending in (0, lnR).
MINIMAX_TEST(find_eta_zeros_locates_sign_changes) {
    const int nlap = 3;
    const double R = 100.0;
    // params that genuinely oscillate around 0 (subtract-off of e^{-t}).
    double a[3] = {0.3, 1.0, 5.0};
    double w[3] = {0.8, 1.0, 0.8};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }
    std::vector<double> z = findEtaZeros(R, nlap, aDD.data(), wDD.data());
    MINIMAX_REQUIRE(!z.empty());
    const double b = std::log(R);
    for (size_t i = 0; i < z.size(); ++i) {
        MINIMAX_REQUIRE(z[i] > 0.0 && z[i] < b);
        if (i) MINIMAX_REQUIRE(z[i] > z[i - 1]);
        MINIMAX_REQUIRE(std::fabs(etaD(z[i], nlap, a, w)) < 1e-9);
    }
}

// Split rule integrates |eta|^n to the independent reference for odd n.
MINIMAX_TEST(build_split_rule_resolves_abs_integral) {
    const int nlap = 3, n = 3;
    const double R = 100.0;
    double a[3] = {0.5, 2.5, 12.0};
    double w[3] = {0.45, 0.55, 1.2};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }

    QuadRule split = buildSplitRule(R, nlap, n, aDD.data(), wDD.data());
    // Weights sum to interval length.
    double sw = 0.0; for (double ww : split.w) sw += ww;
    MINIMAX_REQUIRE_CLOSE(sw, std::log(R), 1e-10);

    double ref = simpsonLnAbs(R, nlap, n, a, w);
    DD Lsplit = evalLnLossAbs(split, nlap, n, aDD.data(), wDD.data(), nullptr, nullptr);
    MINIMAX_REQUIRE_CLOSE(Lsplit.hi, ref, 1e-8);

    // Split rule beats a blind composite GL of comparable node count.
    int panels = static_cast<int>(split.t.size()) / 16;
    QuadRule blind = compositeGaussLegendre(0.0, std::log(R), panels);
    DD Lblind = evalLnLossAbs(blind, nlap, n, aDD.data(), wDD.data(), nullptr, nullptr);
    MINIMAX_REQUIRE(std::fabs(Lsplit.hi - ref) <= std::fabs(Lblind.hi - ref));
}

// Independent composite-Simpson reference for ∫ sqrt(eta^2 + eps^2) e^t dt (x-space measure).
static double simpsonL1Smoothed(double R, int nlap, double eps, const double* a, const double* w) {
    const int M = 200000;               // even
    const double h = std::log(R) / M;
    double acc = 0.0;
    for (int i = 0; i <= M; ++i) {
        double t = i * h;
        double e = etaD(t, nlap, a, w);
        double f = std::sqrt(e * e + eps * eps) * std::exp(t);
        double c = (i == 0 || i == M) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        acc += c * f;
    }
    return acc * h / 3.0;
}

MINIMAX_TEST(l1_smoothed_value_matches_independent_integration) {
    const int nlap = 3;
    const double R = 50.0;
    const double eps = 1e-2;
    double a[3] = {0.5, 2.0, 8.0};
    double w[3] = {0.4, 0.6, 1.5};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    DD L = evalL1LossSmoothed(rule, nlap, DD(eps), aDD.data(), wDD.data(), nullptr, nullptr);
    double ref = simpsonL1Smoothed(R, nlap, eps, a, w);
    MINIMAX_REQUIRE_CLOSE(L.hi, ref, 1e-6);
}

MINIMAX_TEST(l1_smoothed_gradient_matches_finite_difference) {
    const int nlap = 2;
    const double R = 30.0;
    const DD eps(1e-2);
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    std::vector<DD> gA(nlap), gW(nlap);
    evalL1LossSmoothed(rule, nlap, eps, a.data(), w.data(), gA.data(), gW.data());
    const double e = 1e-6;
    for (int k = 0; k < nlap; ++k) {
        std::vector<DD> ap = a; ap[k] = DD(a[k].hi + e);
        std::vector<DD> am = a; am[k] = DD(a[k].hi - e);
        DD Lp = evalL1LossSmoothed(rule, nlap, eps, ap.data(), w.data(), nullptr, nullptr);
        DD Lm = evalL1LossSmoothed(rule, nlap, eps, am.data(), w.data(), nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gA[k].hi, (Lp.hi - Lm.hi) / (2 * e), 1e-4);
        std::vector<DD> wp = w; wp[k] = DD(w[k].hi + e);
        std::vector<DD> wm = w; wm[k] = DD(w[k].hi - e);
        DD Lwp = evalL1LossSmoothed(rule, nlap, eps, a.data(), wp.data(), nullptr, nullptr);
        DD Lwm = evalL1LossSmoothed(rule, nlap, eps, a.data(), wm.data(), nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gW[k].hi, (Lwp.hi - Lwm.hi) / (2 * e), 1e-4);
    }
}

MINIMAX_TEST(l1_smoothed_hessian_matches_finite_difference) {
    const int nlap = 2;
    const double R = 30.0;
    const DD eps(1e-2);
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    evalL1LossSmoothed(rule, nlap, eps, a.data(), w.data(), gA.data(), gW.data(), H.data());
    const double e = 1e-5;
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + e); am[j] = DD(a[j].hi - e); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + e);
                        wm[j - nlap] = DD(w[j - nlap].hi - e); }
        std::vector<DD> gpA(nlap), gpW(nlap), gmA(nlap), gmW(nlap);
        evalL1LossSmoothed(rule, nlap, eps, ap.data(), wp.data(), gpA.data(), gpW.data());
        evalL1LossSmoothed(rule, nlap, eps, am.data(), wm.data(), gmA.data(), gmW.data());
        for (int i = 0; i < dim; ++i) {
            double gp = (i < nlap) ? gpA[i].hi : gpW[i - nlap].hi;
            double gm = (i < nlap) ? gmA[i].hi : gmW[i - nlap].hi;
            MINIMAX_REQUIRE_CLOSE(H[i + j * dim].hi, (gp - gm) / (2 * e), 5e-4);
        }
    }
}

// E1 spot values vs standard references (double precision, ~15 digits).
MINIMAX_TEST(expint1_matches_known_values) {
    struct { double z, e1; } cases[] = {
        {0.5,  0.5597735947761608},
        {1.0,  0.21938393439552027},
        {2.0,  0.04890051070806112},
        {5.0,  1.1482955912753257e-03},
        {10.0, 4.156968929685325e-06},
    };
    for (auto c : cases) {
        DD got = ddExpInt1(DD(c.z));
        MINIMAX_REQUIRE_CLOSE(got.hi, c.e1, 1e-11);
    }
}

// The two independent kernels (Taylor series and modified-Lentz continued fraction) are
// mathematically unrelated algorithms, so agreement to double-double precision is strong
// evidence both are correct. With the exact gamma constant and the CF converged (cap 500),
// they agree to ~3e-29 over z in {1,1.5,2}; the 1e-27 gate leaves ~30x margin.
MINIMAX_TEST(expint1_series_and_cf_agree_to_dd) {
    for (double z : {1.0, 1.5, 2.0}) {
        DD s = ddExpInt1Series(DD(z));
        DD f = ddExpInt1CF(DD(z));
        DD diff = s - f;
        MINIMAX_REQUIRE(std::fabs(diff.hi) < 1e-27 * std::fabs(f.hi));
    }
}

// Deep tail: E1 underflows to exactly 0 once e^{-z} underflows the exponent range.
MINIMAX_TEST(expint1_underflows_to_zero) {
    DD got = ddExpInt1(DD(800.0));
    MINIMAX_REQUIRE(got.hi == 0.0);
}

// Analytic L2 value matches a well-resolved numerical evalLnLoss (n=2).
MINIMAX_TEST(l2_analytic_value_matches_numerical) {
    const int nlap = 3;
    const double R = 80.0;
    double a[3] = {0.5, 2.0, 8.0};
    double w[3] = {0.4, 0.6, 1.5};
    std::vector<DD> aDD(nlap), wDD(nlap);
    for (int k = 0; k < nlap; ++k) { aDD[k] = DD(a[k]); wDD[k] = DD(w[k]); }
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    DD num = evalLnLoss(rule, nlap, 2, aDD.data(), wDD.data(), nullptr, nullptr);
    DD ana = evalL2LossAnalytic(nlap, R, aDD.data(), wDD.data(), nullptr, nullptr);
    MINIMAX_REQUIRE_CLOSE(ana.hi, num.hi, 1e-11);
}

// Analytic gradient vs central finite difference of the analytic value.
MINIMAX_TEST(l2_analytic_gradient_matches_finite_difference) {
    const int nlap = 2;
    const double R = 40.0;
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    std::vector<DD> gA(nlap), gW(nlap);
    evalL2LossAnalytic(nlap, R, a.data(), w.data(), gA.data(), gW.data());
    const double eps = 1e-6;
    for (int k = 0; k < nlap; ++k) {
        std::vector<DD> ap = a; ap[k] = DD(a[k].hi + eps);
        std::vector<DD> am = a; am[k] = DD(a[k].hi - eps);
        DD Lp = evalL2LossAnalytic(nlap, R, ap.data(), w.data(), nullptr, nullptr);
        DD Lm = evalL2LossAnalytic(nlap, R, am.data(), w.data(), nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gA[k].hi, (Lp.hi - Lm.hi) / (2 * eps), 1e-6);
        std::vector<DD> wp = w; wp[k] = DD(w[k].hi + eps);
        std::vector<DD> wm = w; wm[k] = DD(w[k].hi - eps);
        DD Lwp = evalL2LossAnalytic(nlap, R, a.data(), wp.data(), nullptr, nullptr);
        DD Lwm = evalL2LossAnalytic(nlap, R, a.data(), wm.data(), nullptr, nullptr);
        MINIMAX_REQUIRE_CLOSE(gW[k].hi, (Lwp.hi - Lwm.hi) / (2 * eps), 1e-6);
    }
}

// Analytic gradient agrees with the numerical evalLnLoss gradient (n=2).
MINIMAX_TEST(l2_analytic_gradient_matches_numerical) {
    const int nlap = 3;
    const double R = 80.0;
    std::vector<DD> a(nlap), w(nlap);
    double av[3] = {0.5, 2.0, 8.0}, wv[3] = {0.4, 0.6, 1.5};
    for (int k = 0; k < nlap; ++k) { a[k] = DD(av[k]); w[k] = DD(wv[k]); }
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    std::vector<DD> gAn(nlap), gWn(nlap), gAa(nlap), gWa(nlap);
    evalLnLoss(rule, nlap, 2, a.data(), w.data(), gAn.data(), gWn.data());
    evalL2LossAnalytic(nlap, R, a.data(), w.data(), gAa.data(), gWa.data());
    for (int k = 0; k < nlap; ++k) {
        MINIMAX_REQUIRE_CLOSE(gAa[k].hi, gAn[k].hi, 1e-9);
        MINIMAX_REQUIRE_CLOSE(gWa[k].hi, gWn[k].hi, 1e-9);
    }
}

// Analytic Hessian vs central finite difference of the analytic gradient.
MINIMAX_TEST(l2_analytic_hessian_matches_finite_difference) {
    const int nlap = 2;
    const double R = 40.0;
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap);
    a[0] = DD(0.7); a[1] = DD(3.0); w[0] = DD(0.5); w[1] = DD(0.9);
    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    evalL2LossAnalytic(nlap, R, a.data(), w.data(), gA.data(), gW.data(), H.data());
    const double eps = 1e-5;
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else          { wp[j - nlap] = DD(w[j - nlap].hi + eps);
                        wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        std::vector<DD> gpA(nlap), gpW(nlap), gmA(nlap), gmW(nlap);
        evalL2LossAnalytic(nlap, R, ap.data(), wp.data(), gpA.data(), gpW.data());
        evalL2LossAnalytic(nlap, R, am.data(), wm.data(), gmA.data(), gmW.data());
        for (int i = 0; i < dim; ++i) {
            double gp = (i < nlap) ? gpA[i].hi : gpW[i - nlap].hi;
            double gm = (i < nlap) ? gmA[i].hi : gmW[i - nlap].hi;
            MINIMAX_REQUIRE_CLOSE(H[i + j * dim].hi, (gp - gm) / (2 * eps), 5e-5);
        }
    }
}

// Analytic Hessian agrees with the numerical evalLnLoss Hessian (n=2).
MINIMAX_TEST(l2_analytic_hessian_matches_numerical) {
    const int nlap = 3;
    const double R = 80.0;
    const int dim = 2 * nlap;
    std::vector<DD> a(nlap), w(nlap);
    double av[3] = {0.5, 2.0, 8.0}, wv[3] = {0.4, 0.6, 1.5};
    for (int k = 0; k < nlap; ++k) { a[k] = DD(av[k]); w[k] = DD(wv[k]); }
    QuadRule rule = compositeGaussLegendre(0.0, std::log(R), 8192);
    std::vector<DD> gA(nlap), gW(nlap);
    std::vector<DD> Hn(static_cast<size_t>(dim) * dim), Ha(static_cast<size_t>(dim) * dim);
    evalLnLoss(rule, nlap, 2, a.data(), w.data(), gA.data(), gW.data(), Hn.data());
    evalL2LossAnalytic(nlap, R, a.data(), w.data(), gA.data(), gW.data(), Ha.data());
    for (int i = 0; i < dim * dim; ++i)
        MINIMAX_REQUIRE_CLOSE(Ha[i].hi, Hn[i].hi, 1e-8);
}

// A config whose eta oscillates (so it has several well-separated interior
// zeros) is exactly the minimax solution: it equioscillates by construction.
static void minimaxConfig(int nlap, double R, std::vector<DD>& a, std::vector<DD>& w) {
    auto mm = minimax_cpppy::detail::laplaceMinimax(nlap, 1.0, R, 200, 1e-10,
                                                    1e-15, 0.3, 1e-6, 1e-4, 0, nullptr);
    a.resize(nlap); w.resize(nlap);
    for (int k = 0; k < nlap; ++k) { a[k] = DD(mm.expon[k]); w[k] = DD(mm.weight[k]); }
}

// Analytic L1 value equals a well-resolved numerical |eta| integral (n=1) on a
// split-at-zeros composite GL rule (kinks land on panel boundaries).
MINIMAX_TEST(l1_analytic_value_matches_numerical) {
    const int nlap = 4;
    const double R = 100.0;
    std::vector<DD> a, w;
    minimaxConfig(nlap, R, a, w);
    std::vector<double> zeros = findEtaZeros(R, nlap, a.data(), w.data());
    MINIMAX_REQUIRE(zeros.size() > 0);   // config must actually oscillate

    QuadRule split = buildSplitRule(R, nlap, 1.0, a.data(), w.data(), zeros);
    DD num = evalLnLossAbs(split, nlap, 1.0, a.data(), w.data(), nullptr, nullptr);
    DD ana = evalL1LossAnalytic(nlap, R, a.data(), w.data(),
                                nullptr, nullptr, nullptr, zeros);
    MINIMAX_REQUIRE_CLOSE(ana.hi, num.hi, 1e-10);
}

// Analytic gradient vs central FD of the analytic value. Each perturbed value
// recomputes its OWN zeros (the value depends on them as breakpoints); this is
// what validates that the moving-zero boundary terms vanish from the gradient.
MINIMAX_TEST(l1_analytic_gradient_matches_finite_difference) {
    const int nlap = 4;
    const double R = 100.0;
    std::vector<DD> a, w;
    minimaxConfig(nlap, R, a, w);
    std::vector<double> z0 = findEtaZeros(R, nlap, a.data(), w.data());

    std::vector<DD> gA(nlap), gW(nlap);
    evalL1LossAnalytic(nlap, R, a.data(), w.data(), gA.data(), gW.data(), nullptr, z0);

    auto valAt = [&](const std::vector<DD>& aa, const std::vector<DD>& ww) {
        std::vector<double> z = findEtaZeros(R, nlap, aa.data(), ww.data());
        return evalL1LossAnalytic(nlap, R, aa.data(), ww.data(),
                                  nullptr, nullptr, nullptr, z).hi;
    };
    const double eps = 1e-6;
    for (int k = 0; k < nlap; ++k) {
        std::vector<DD> ap = a; ap[k] = DD(a[k].hi + eps);
        std::vector<DD> am = a; am[k] = DD(a[k].hi - eps);
        MINIMAX_REQUIRE_CLOSE(gA[k].hi, (valAt(ap, w) - valAt(am, w)) / (2 * eps), 1e-6);
        std::vector<DD> wp = w; wp[k] = DD(w[k].hi + eps);
        std::vector<DD> wm = w; wm[k] = DD(w[k].hi - eps);
        MINIMAX_REQUIRE_CLOSE(gW[k].hi, (valAt(a, wp) - valAt(a, wm)) / (2 * eps), 1e-6);
    }
}

// Analytic Hessian vs central FD of the analytic GRADIENT. Each perturbed
// gradient recomputes its own zeros; the rank-1 corrections in the base
// Hessian are exactly the curvature FD picks up from the zeros moving.
MINIMAX_TEST(l1_analytic_hessian_matches_finite_difference) {
    const int nlap = 4;
    const double R = 100.0;
    const int dim = 2 * nlap;
    std::vector<DD> a, w;
    minimaxConfig(nlap, R, a, w);
    std::vector<double> z0 = findEtaZeros(R, nlap, a.data(), w.data());

    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    bool ok = false;
    evalL1LossAnalytic(nlap, R, a.data(), w.data(),
                       gA.data(), gW.data(), H.data(), z0, &ok);
    MINIMAX_REQUIRE(ok);

    auto gradAt = [&](const std::vector<DD>& aa, const std::vector<DD>& ww,
                      std::vector<DD>& gAo, std::vector<DD>& gWo) {
        std::vector<double> z = findEtaZeros(R, nlap, aa.data(), ww.data());
        evalL1LossAnalytic(nlap, R, aa.data(), ww.data(),
                           gAo.data(), gWo.data(), nullptr, z);
    };
    const double eps = 1e-6;
    for (int j = 0; j < dim; ++j) {
        std::vector<DD> ap = a, wp = w, am = a, wm = w;
        if (j < nlap) { ap[j] = DD(a[j].hi + eps); am[j] = DD(a[j].hi - eps); }
        else { wp[j - nlap] = DD(w[j - nlap].hi + eps);
               wm[j - nlap] = DD(w[j - nlap].hi - eps); }
        std::vector<DD> gpA(nlap), gpW(nlap), gmA(nlap), gmW(nlap);
        gradAt(ap, wp, gpA, gpW);
        gradAt(am, wm, gmA, gmW);
        for (int i = 0; i < dim; ++i) {
            double gp = (i < nlap) ? gpA[i].hi : gpW[i - nlap].hi;
            double gm = (i < nlap) ? gmA[i].hi : gmW[i - nlap].hi;
            MINIMAX_REQUIRE_CLOSE(H[i + j * dim].hi, (gp - gm) / (2 * eps), 1e-5);
        }
    }
}

// Each interior-zero rank-1 coefficient c_j = -2 s_{j-1} x_j / eta_t(t_j) > 0.
// Recompute it from public pieces (guards the sign derivation independently).
MINIMAX_TEST(l1_analytic_rank1_coeffs_positive) {
    const int nlap = 4;
    const double R = 100.0;
    std::vector<DD> a, w;
    minimaxConfig(nlap, R, a, w);
    std::vector<double> zeros = findEtaZeros(R, nlap, a.data(), w.data());
    MINIMAX_REQUIRE(zeros.size() > 0);

    DD eta0 = DD(1.0);
    for (int k = 0; k < nlap; ++k)
        eta0 = eta0 - w[k] * DD::ddExp(-(a[k] * DD::ddExp(DD(0.0))));
    const double s0 = (eta0.hi < 0.0) ? -1.0 : 1.0;
    for (size_t i = 0; i < zeros.size(); ++i) {
        DD eta, etaP;
        etaAndPrimeAt(zeros[i], nlap, a.data(), w.data(), eta, etaP);
        const double sjm1 = ((static_cast<int>(i) % 2) == 0) ? s0 : -s0; // s_{j-1}, j=i+1
        const double xj = DD::ddExp(DD(zeros[i])).hi;
        const double cj = -2.0 * sjm1 * xj / etaP.hi;
        MINIMAX_REQUIRE(cj > 0.0);
    }
}

// Near-tangent zero -> rank-1 blows up -> *ok=false. Constructed by passing an
// interior EXTREMUM of eta (where eta_t=0) as a fake "zero".
MINIMAX_TEST(l1_analytic_near_tangent_sets_ok_false) {
    const int nlap = 4;
    const double R = 100.0;
    std::vector<DD> a, w;
    minimaxConfig(nlap, R, a, w);
    const double b = std::log(R);

    // Bracket a sign change of eta_t on (0,b), then bisect to an extremum.
    const int M = 4000;
    double tstar = -1.0;
    DD e, ep; etaAndPrimeAt(0.0, nlap, a.data(), w.data(), e, ep);
    double fprev = ep.hi, tprev = 0.0;
    for (int i = 1; i <= M && tstar < 0.0; ++i) {
        double t = b * i / M;
        etaAndPrimeAt(t, nlap, a.data(), w.data(), e, ep);
        if (fprev != 0.0 && (fprev < 0.0) != (ep.hi < 0.0)) {
            double lo = tprev, hi = t, flo = fprev;
            for (int it = 0; it < 100; ++it) {
                double mid = 0.5 * (lo + hi);
                etaAndPrimeAt(mid, nlap, a.data(), w.data(), e, ep);
                if ((ep.hi < 0.0) == (flo < 0.0)) { lo = mid; flo = ep.hi; } else hi = mid;
            }
            tstar = 0.5 * (lo + hi);
        }
        fprev = ep.hi; tprev = t;
    }
    MINIMAX_REQUIRE(tstar > 0.0);   // eta must have an interior extremum

    const int dim = 2 * nlap;
    std::vector<DD> gA(nlap), gW(nlap), H(static_cast<size_t>(dim) * dim);
    std::vector<double> fakeZeros = { tstar };
    bool ok = true;
    evalL1LossAnalytic(nlap, R, a.data(), w.data(),
                       gA.data(), gW.data(), H.data(), fakeZeros, &ok);
    MINIMAX_REQUIRE(!ok);
}

// findEtaZerosWarm must detect when the tracked zero set is INCOMPLETE (the
// true zero count grew since prevZeros was captured -- e.g. a large outer
// Newton step revealed a new zero pair near the low-t end) and fall back to
// the full DD scan rather than silently returning the short, self-consistent
// list. Simulate the incomplete-tracked-list case directly: drop the
// lowest-t zeros from the true full-scan set and hand the remainder to
// findEtaZerosWarm as "prevZeros".
MINIMAX_TEST(find_eta_zeros_warm_recovers_from_incomplete_tracked_set) {
    const int nlap = 4;
    const double R = 100.0;
    std::vector<DD> a, w;
    minimaxConfig(nlap, R, a, w);

    std::vector<double> zFull = findEtaZeros(R, nlap, a.data(), w.data());
    MINIMAX_REQUIRE(zFull.size() >= 3);   // need at least a couple to drop

    // Drop the two lowest-t zeros -- what a warm start would look like if
    // Newton's early steps hadn't yet revealed them.
    std::vector<double> prevShort(zFull.begin() + 2, zFull.end());

    std::vector<double> zWarm = findEtaZerosWarm(R, nlap, a.data(), w.data(), prevShort);

    MINIMAX_REQUIRE(zWarm.size() == zFull.size());
    for (size_t i = 0; i < zFull.size(); ++i)
        MINIMAX_REQUIRE_CLOSE(zWarm[i], zFull[i], 1e-12);
}

// When the tracked set IS complete (count matches), findEtaZerosWarm must
// still return the (cheaply refined) warm result unchanged -- the
// completeness check must not force an unnecessary full-scan fallback.
MINIMAX_TEST(find_eta_zeros_warm_keeps_warm_result_when_complete) {
    const int nlap = 4;
    const double R = 100.0;
    std::vector<DD> a, w;
    minimaxConfig(nlap, R, a, w);

    std::vector<double> zFull = findEtaZeros(R, nlap, a.data(), w.data());
    MINIMAX_REQUIRE(zFull.size() > 0);

    // Same config -> the "warm" refinement from the true zeros themselves is
    // a no-op fixed point; the coarse count must match n and return promptly.
    std::vector<double> zWarm = findEtaZerosWarm(R, nlap, a.data(), w.data(), zFull);

    MINIMAX_REQUIRE(zWarm.size() == zFull.size());
    for (size_t i = 0; i < zFull.size(); ++i)
        MINIMAX_REQUIRE_CLOSE(zWarm[i], zFull[i], 1e-12);
}

int main() { MINIMAX_RUN_TESTS(); }
