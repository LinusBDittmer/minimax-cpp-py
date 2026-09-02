// tests/test_minimax_lp.cpp
#include "test_helpers.hpp"
#include "minimax_cpppy/minimax.hpp"
#include "minimax_cpppy/minimax_lp.hpp"
#include "core/algorithm.hpp"
#include "core/ln_loss.hpp"
#include <cmath>

using minimax_cpppy::laplaceMinimax;
using minimax_cpppy::laplaceLp;
using minimax_cpppy::MinimaxResult;
using minimax_cpppy::detail::DD;
using minimax_cpppy::detail::QuadRule;
using minimax_cpppy::detail::compositeGaussLegendre;
using minimax_cpppy::detail::evalLnLoss;
using minimax_cpppy::detail::buildLnRule;

// Returns nlap exponents/weights, all finite, positive exponents.
MINIMAX_TEST(ln_api_basic_shape) {
    MinimaxResult r = laplaceLp(4, 1.0, 100.0, 4, 0, std::cerr);
    MINIMAX_REQUIRE(static_cast<int>(r.expon.size()) == 4);
    MINIMAX_REQUIRE(static_cast<int>(r.weight.size()) == 4);
    for (int k = 0; k < 4; ++k) {
        MINIMAX_REQUIRE(std::isfinite(r.expon[k]) && r.expon[k] > 0.0);
        MINIMAX_REQUIRE(std::isfinite(r.weight[k]));
    }
    MINIMAX_REQUIRE(std::isfinite(r.errmax) && r.errmax > 0.0);
}

// L_n optimum is <= minimax solution measured in the L_n norm.
MINIMAX_TEST(ln_beats_minimax_in_Ln) {
    const int nlap = 4, n = 4;
    const double ymin = 1.0, ymax = 200.0;
    MinimaxResult mm = laplaceMinimax(nlap, ymin, ymax);
    MinimaxResult ln = laplaceLp(nlap, ymin, ymax, n, 0, std::cerr);
    // recompute the minimax solution's L_n norm via the same engine path
    double R = ymax / ymin;
    // crude L_n norm of mm via dense Simpson on the physical error
    auto etaP = [&](double y, const MinimaxResult& r) {
        double s = 1.0 / y;
        for (int k = 0; k < nlap; ++k) s -= r.weight[k] * std::exp(-r.expon[k] * y);
        return s;
    };
    auto lnNorm = [&](const MinimaxResult& r) {
        const int M = 100000; double h = std::log(R) / M, acc = 0.0;
        for (int i = 0; i <= M; ++i) {
            double t = i * h, y = ymin * std::exp(t);
            double f = std::pow(etaP(y, r), n) * y;   // x-space measure: dy = y dt
            double c = (i == 0 || i == M) ? 1.0 : (i % 2 ? 4.0 : 2.0);
            acc += c * f;
        }
        return std::pow(acc * h / 3.0, 1.0 / n);
    };
    MINIMAX_REQUIRE(lnNorm(ln) <= lnNorm(mm) * (1.0 + 1e-6));
}

// nlap=1 works.
MINIMAX_TEST(ln_api_nlap_one) {
    MinimaxResult r = laplaceLp(1, 1.0, 10.0, 2, 0, std::cerr);
    MINIMAX_REQUIRE(static_cast<int>(r.expon.size()) == 1);
    MINIMAX_REQUIRE(std::isfinite(r.errmax) && r.errmax > 0.0);
}

// Larger n moves the L_inf error of the result toward the minimax error.
MINIMAX_TEST(ln_large_n_approaches_minimax_Linf) {
    const int nlap = 4;
    const double ymin = 1.0, ymax = 100.0;
    MinimaxResult mm = laplaceMinimax(nlap, ymin, ymax);
    auto linf = [&](const MinimaxResult& r) {
        double m = 0.0;
        for (int i = 0; i <= 4000; ++i) {
            double y = ymin * std::pow(ymax / ymin, i / 4000.0);
            double s = 1.0 / y;
            for (int k = 0; k < nlap; ++k) s -= r.weight[k] * std::exp(-r.expon[k] * y);
            m = std::max(m, std::abs(s));
        }
        return m;
    };
    double e2  = linf(laplaceLp(nlap, ymin, ymax, 2,  0, std::cerr));
    double e12 = linf(laplaceLp(nlap, ymin, ymax, 12, 0, std::cerr));
    double emm = linf(mm);
    // higher n is closer to the minimax L_inf error
    MINIMAX_REQUIRE(std::abs(e12 - emm) <= std::abs(e2 - emm) + 1e-12);
}

// Increasing the norm order n drives the L_n points/weights toward the minimax
// (L_inf) solution. Across representative (R, nlap) the highest order recovers
// minimax to machine precision, while n=2 (least squares) leaves a real gap.
MINIMAX_TEST(ln_increasing_order_converges_to_minimax) {
    struct Case { double R; int nlap; };
    const Case cases[] = {{10.0, 2}, {100.0, 3}, {1000.0, 4}};
    const int ns[] = {2, 4, 6, 8};
    const int NN = 4;

    // Max relative difference of exponents/weights between two solutions.
    auto paramDist = [](const MinimaxResult& a, const MinimaxResult& b, int nlap) {
        double m = 0.0;
        for (int k = 0; k < nlap; ++k) {
            double de = std::abs(a.expon[k]  - b.expon[k])  / std::max(1e-300, std::abs(b.expon[k]));
            double dw = std::abs(a.weight[k] - b.weight[k]) / std::max(1e-300, std::abs(b.weight[k]));
            m = std::max(m, std::max(de, dw));
        }
        return m;
    };

    for (const Case& c : cases) {
        MinimaxResult mm = laplaceMinimax(c.nlap, 1.0, c.R);
        double d[NN];
        for (int i = 0; i < NN; ++i) {
            MinimaxResult ln = laplaceLp(c.nlap, 1.0, c.R, ns[i], 0, std::cerr);
            d[i] = paramDist(ln, mm, c.nlap);
        }
        // n=2 (least squares) sits a real distance from the minimax solution.
        MINIMAX_REQUIRE(d[0] > 1e-3);
        // Increasing the order converges to minimax: the highest order is far
        // closer than n=2 and is essentially the minimax solution itself.
        MINIMAX_REQUIRE(d[NN - 1] < d[0]);
        MINIMAX_REQUIRE(d[NN - 1] < 1e-9);
        // Monotone non-increasing from n=4 onward (the n=2 -> n=4 step may bump
        // slightly for some R/nlap, so it is not asserted as monotone).
        for (int i = 1; i + 1 < NN; ++i)
            MINIMAX_REQUIRE(d[i + 1] <= d[i] + 1e-12);
    }
}

// n=1 runs and returns a sane, finite result.
MINIMAX_TEST(ln_n1_returns_finite_result) {
    auto r = minimax_cpppy::laplaceLp(4, 1.0, 100.0, 1);
    MINIMAX_REQUIRE(static_cast<int>(r.expon.size()) == 4);
    for (int k = 0; k < 4; ++k) {
        MINIMAX_REQUIRE(std::isfinite(r.expon[k]) && r.expon[k] > 0.0);
        MINIMAX_REQUIRE(std::isfinite(r.weight[k]));
    }
    MINIMAX_REQUIRE(std::isfinite(r.errmax) && r.errmax > 0.0);
}

// n=1 optimiser strictly improves the L1 loss vs the minimax warm-start.
MINIMAX_TEST(ln_n1_beats_minimax_init_on_L1) {
    const int nlap = 4, n = 1;
    const double R = 100.0;
    using minimax_cpppy::detail::DD;
    // minimax warm-start (ymin=1 => normalised == physical).
    auto mm = minimax_cpppy::detail::laplaceMinimax(nlap, 1.0, R, 200, 1e-10,
                                                    1e-15, 0.3, 1e-6, 1e-4, 0, nullptr);
    std::vector<DD> a0(nlap), w0(nlap);
    for (int k = 0; k < nlap; ++k) { a0[k] = DD(mm.expon[k]); w0[k] = DD(mm.weight[k]); }
    auto ruleInit = minimax_cpppy::detail::buildSplitRule(R, nlap, n, a0.data(), w0.data());
    DD L1init = minimax_cpppy::detail::evalLnLossAbs(ruleInit, nlap, n,
                                                     a0.data(), w0.data(), nullptr, nullptr);

    auto r = minimax_cpppy::laplaceLp(nlap, 1.0, R, n);
    std::vector<DD> ar(nlap), wr(nlap);
    for (int k = 0; k < nlap; ++k) { ar[k] = DD(r.expon[k]); wr[k] = DD(r.weight[k]); }
    auto ruleRes = minimax_cpppy::detail::buildSplitRule(R, nlap, n, ar.data(), wr.data());
    DD L1res = minimax_cpppy::detail::evalLnLossAbs(ruleRes, nlap, n,
                                                    ar.data(), wr.data(), nullptr, nullptr);
    MINIMAX_REQUIRE(L1res.hi <= L1init.hi * (1.0 + 1e-9));
}

// odd n: large n approaches the minimax (L_inf) error.
MINIMAX_TEST(ln_odd_large_n_approaches_minimax) {
    const int nlap = 4;
    const double R = 100.0;
    auto mm = minimax_cpppy::detail::laplaceMinimax(nlap, 1.0, R, 200, 1e-10,
                                                    1e-15, 0.3, 1e-6, 1e-4, 0, nullptr);
    auto r = minimax_cpppy::laplaceLp(nlap, 1.0, R, 15);
    // Large n drives the result's L_inf error toward the minimax L_inf error.
    // (Compare L_inf directly: the reported errmax is now the x-space L_n norm,
    // which carries the e^t weight and only equals max|eta| in the n->inf limit.)
    double m = 0.0;
    for (int i = 0; i <= 4000; ++i) {
        double y = std::pow(R, i / 4000.0);
        double s = 1.0 / y;
        for (int k = 0; k < nlap; ++k) s -= r.weight[k] * std::exp(-r.expon[k] * y);
        m = std::max(m, std::abs(s));
    }
    MINIMAX_REQUIRE_CLOSE(m, mm.errmax, 0.05);
}

// even-n path still works (untouched dispatch branch).
MINIMAX_TEST(ln_even_n_still_converges) {
    auto r = minimax_cpppy::laplaceLp(4, 1.0, 100.0, 4);
    MINIMAX_REQUIRE(static_cast<int>(r.expon.size()) == 4);
    for (int k = 0; k < 4; ++k)
        MINIMAX_REQUIRE(std::isfinite(r.expon[k]) && r.expon[k] > 0.0);
    MINIMAX_REQUIRE(std::isfinite(r.errmax) && r.errmax > 0.0);
}

// non-integer norm order (n=2.5): runs, finite, and the optimum is <= the
// minimax solution measured in the same real-order L_n norm.
MINIMAX_TEST(ln_noninteger_n_beats_minimax) {
    const int nlap = 4;
    const double n = 2.5, ymin = 1.0, ymax = 200.0;
    MinimaxResult mm = laplaceMinimax(nlap, ymin, ymax);
    MinimaxResult ln = laplaceLp(nlap, ymin, ymax, n, 0, std::cerr);
    for (int k = 0; k < nlap; ++k) {
        MINIMAX_REQUIRE(std::isfinite(ln.expon[k]) && ln.expon[k] > 0.0);
        MINIMAX_REQUIRE(std::isfinite(ln.weight[k]));
    }
    MINIMAX_REQUIRE(std::isfinite(ln.errmax) && ln.errmax > 0.0);
    const double R = ymax / ymin;
    auto etaP = [&](double y, const MinimaxResult& r) {
        double s = 1.0 / y;
        for (int k = 0; k < nlap; ++k) s -= r.weight[k] * std::exp(-r.expon[k] * y);
        return s;
    };
    auto lnNorm = [&](const MinimaxResult& r) {
        const int M = 100000; double h = std::log(R) / M, acc = 0.0;
        for (int i = 0; i <= M; ++i) {
            double t = i * h, y = ymin * std::exp(t);
            double f = std::pow(std::abs(etaP(y, r)), n) * y;   // x-space measure: dy = y dt
            double c = (i == 0 || i == M) ? 1.0 : (i % 2 ? 4.0 : 2.0);
            acc += c * f;
        }
        return std::pow(acc * h / 3.0, 1.0 / n);
    };
    MINIMAX_REQUIRE(lnNorm(ln) <= lnNorm(mm) * (1.0 + 1e-6));
}

// Low non-integer order (n=1.2): direct minimax-warm-started Newton stalls on the
// singular |eta|^{n-2} Hessian, so the continuation-in-n fallback must kick in and
// still return a finite, optimal solution. Small config to keep the test affordable.
MINIMAX_TEST(ln_low_noninteger_n_continuation) {
    const int nlap = 2;
    const double n = 1.2, ymin = 1.0, ymax = 10.0;
    MinimaxResult mm = laplaceMinimax(nlap, ymin, ymax);
    MinimaxResult ln = laplaceLp(nlap, ymin, ymax, n, 0, std::cerr);
    for (int k = 0; k < nlap; ++k) {
        MINIMAX_REQUIRE(std::isfinite(ln.expon[k]) && ln.expon[k] > 0.0);
        MINIMAX_REQUIRE(std::isfinite(ln.weight[k]));
    }
    MINIMAX_REQUIRE(std::isfinite(ln.errmax) && ln.errmax > 0.0);
    // The continuation result must be a real optimum: <= minimax in the L_n norm.
    const double R = ymax / ymin;
    auto etaP = [&](double y, const MinimaxResult& r) {
        double s = 1.0 / y;
        for (int k = 0; k < nlap; ++k) s -= r.weight[k] * std::exp(-r.expon[k] * y);
        return s;
    };
    auto lnNorm = [&](const MinimaxResult& r) {
        const int M = 100000; double h = std::log(R) / M, acc = 0.0;
        for (int i = 0; i <= M; ++i) {
            double t = i * h, y = ymin * std::exp(t);
            double f = std::pow(std::abs(etaP(y, r)), n) * y;   // x-space measure: dy = y dt
            double c = (i == 0 || i == M) ? 1.0 : (i % 2 ? 4.0 : 2.0);
            acc += c * f;
        }
        return std::pow(acc * h / 3.0, 1.0 / n);
    };
    MINIMAX_REQUIRE(lnNorm(ln) <= lnNorm(mm) * (1.0 + 1e-6));
}

// The analytic n=2 solution is a genuine optimum of the numerical L2 loss:
// the z=(ln a, w)-space gradient under evalLnLoss is ~0, and the reported
// errmax equals (∫ η^2)^{1/2} measured by the independent numerical engine.
MINIMAX_TEST(ln_n2_analytic_is_optimal_under_numerical_engine) {
    const int nlap = 3;
    const double R = 80.0;
    MinimaxResult r = laplaceLp(nlap, 1.0, R, 2, 0, std::cerr);

    std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap);
    for (int k = 0; k < nlap; ++k) { a[k] = DD(r.expon[k]); w[k] = DD(r.weight[k]); }
    QuadRule rule = buildLnRule(R, nlap, 2, a.data(), w.data());
    DD L = evalLnLoss(rule, nlap, 2, a.data(), w.data(), gA.data(), gW.data());

    // z-space gradient: ∂/∂(ln a_k) = a_k gA_k, ∂/∂w_k = gW_k. ~0 at optimum.
    double gmax = 0.0;
    for (int k = 0; k < nlap; ++k) {
        gmax = std::max(gmax, std::fabs((a[k] * gA[k]).hi));
        gmax = std::max(gmax, std::fabs(gW[k].hi));
    }
    MINIMAX_REQUIRE(gmax < 1e-6);
    MINIMAX_REQUIRE_CLOSE(r.errmax, std::sqrt(L.hi), 1e-8);
}

// The analytic n=1 solution is a genuine L1 optimum: its z=(ln a, w)-space
// gradient under the analytic engine is ~0, and the reported errmax equals the
// L1 value measured independently on a fine split-at-zeros numerical rule.
MINIMAX_TEST(ln_n1_analytic_is_optimal) {
    using minimax_cpppy::detail::DD;
    const int nlap = 4;
    const double R = 100.0;
    MinimaxResult r = laplaceLp(nlap, 1.0, R, 1, 0, std::cerr);

    std::vector<DD> a(nlap), w(nlap), gA(nlap), gW(nlap);
    for (int k = 0; k < nlap; ++k) { a[k] = DD(r.expon[k]); w[k] = DD(r.weight[k]); }
    std::vector<double> zeros =
        minimax_cpppy::detail::findEtaZeros(R, nlap, a.data(), w.data());
    DD L = minimax_cpppy::detail::evalL1LossAnalytic(
        nlap, R, a.data(), w.data(), gA.data(), gW.data(), nullptr, zeros);

    // z-space gradient: ∂/∂(ln a_k)=a_k gA_k, ∂/∂w_k=gW_k. ~0 at optimum.
    double gmax = 0.0;
    for (int k = 0; k < nlap; ++k) {
        gmax = std::max(gmax, std::fabs((a[k] * gA[k]).hi));
        gmax = std::max(gmax, std::fabs(gW[k].hi));
    }
    MINIMAX_REQUIRE(gmax < 1e-7);

    // errmax == L1 (n=1: (L_1)^{1/1}) == independent numerical split-rule L1.
    auto rule = minimax_cpppy::detail::buildSplitRule(R, nlap, 1.0, a.data(), w.data(), zeros);
    DD Lnum = minimax_cpppy::detail::evalLnLossAbs(
        rule, nlap, 1.0, a.data(), w.data(), nullptr, nullptr);
    MINIMAX_REQUIRE_CLOSE(r.errmax, L.hi, 1e-10);
    MINIMAX_REQUIRE_CLOSE(r.errmax, Lnum.hi, 1e-9);
}

int main() { MINIMAX_RUN_TESTS(); }
