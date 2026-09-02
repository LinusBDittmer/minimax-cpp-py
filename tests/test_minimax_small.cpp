#include "test_helpers.hpp"
#include "minimax_cpppy/minimax.hpp"
#include <cmath>
#include <iostream>
#include <vector>

static constexpr double YMIN = 1.0;
static constexpr double YMAX = 100.0;

static double max_approx_error(const minimax_cpppy::MinimaxResult& r,
                                double ymin, double ymax, int ngrid = 1000)
{
    double err = 0.0;
    const int n = static_cast<int>(r.expon.size());
    for (int i = 0; i < ngrid; ++i) {
        const double t = static_cast<double>(i) / (ngrid - 1);
        const double x = ymin * std::pow(ymax / ymin, t);
        double approx = 0.0;
        for (int k = 0; k < n; ++k)
            approx += r.weight[k] * std::exp(-r.expon[k] * x);
        err = std::max(err, std::abs(1.0 / x - approx));
    }
    return err;
}

MINIMAX_TEST(nlap3_runs_and_approximates) {
    auto r = minimax_cpppy::laplaceMinimax(3, YMIN, YMAX);
    MINIMAX_REQUIRE(r.expon.size() == 3);
    MINIMAX_REQUIRE(r.weight.size() == 3);
    const double err = max_approx_error(r, YMIN, YMAX);
    std::cout << "    nlap=3 max error: " << err << "\n";
    MINIMAX_REQUIRE(err < 1e-2);
}

MINIMAX_TEST(nlap5_runs_and_approximates) {
    auto r = minimax_cpppy::laplaceMinimax(5, YMIN, YMAX);
    MINIMAX_REQUIRE(r.expon.size() == 5);
    MINIMAX_REQUIRE(r.weight.size() == 5);
    const double err = max_approx_error(r, YMIN, YMAX);
    std::cout << "    nlap=5 max error: " << err << "\n";
    MINIMAX_REQUIRE(err < 1e-3);
}

MINIMAX_TEST(nlap7_runs_and_approximates) {
    auto r = minimax_cpppy::laplaceMinimax(7, YMIN, YMAX);
    MINIMAX_REQUIRE(r.expon.size() == 7);
    MINIMAX_REQUIRE(r.weight.size() == 7);
    const double err = max_approx_error(r, YMIN, YMAX);
    std::cout << "    nlap=7 max error: " << err << "\n";
    MINIMAX_REQUIRE(err < 1e-4);
}

MINIMAX_TEST(error_decreases_with_nlap) {
    double prev_err = 1e99;
    for (int nlap : {3, 5, 7, 10}) {
        auto r = minimax_cpppy::laplaceMinimax(nlap, YMIN, YMAX);
        const double err = max_approx_error(r, YMIN, YMAX);
        MINIMAX_REQUIRE(err < prev_err);
        prev_err = err;
    }
}

MINIMAX_TEST(errmax_field_is_positive) {
    auto r = minimax_cpppy::laplaceMinimax(5, YMIN, YMAX);
    MINIMAX_REQUIRE(r.errmax > 0.0);
}

MINIMAX_TEST(invalid_nlap_31_throws) {
    MINIMAX_REQUIRE_THROW_TYPE(minimax_cpppy::laplaceMinimax(31, YMIN, YMAX), std::invalid_argument);
}

MINIMAX_TEST(invalid_nlap_0_throws) {
    MINIMAX_REQUIRE_THROW_TYPE(minimax_cpppy::laplaceMinimax(0, YMIN, YMAX), std::invalid_argument);
}

MINIMAX_TEST(invalid_range_throws) {
    MINIMAX_REQUIRE_THROW_TYPE(minimax_cpppy::laplaceMinimax(5, 100.0, 1.0), std::invalid_argument);
    MINIMAX_REQUIRE_THROW_TYPE(minimax_cpppy::laplaceMinimax(5, 1.0, 1.0), std::invalid_argument);
}

MINIMAX_TEST(init_guess_matches_standard_result) {
    auto ref = minimax_cpppy::laplaceMinimax(5, YMIN, YMAX);
    auto r   = minimax_cpppy::laplaceMinimax(5, YMIN, YMAX,
                                              ref.expon, ref.weight);
    MINIMAX_REQUIRE(r.expon.size() == 5);
    MINIMAX_REQUIRE(r.weight.size() == 5);
    const double err = max_approx_error(r, YMIN, YMAX);
    MINIMAX_REQUIRE(err < 1e-3);
}

MINIMAX_TEST(init_guess_wrong_size_throws) {
    std::vector<double> bad_e = {1.0, 2.0};
    std::vector<double> bad_w = {0.1, 0.2};
    MINIMAX_REQUIRE_THROW_TYPE(minimax_cpppy::laplaceMinimax(5, YMIN, YMAX, bad_e, bad_w), std::invalid_argument);
}

MINIMAX_TEST(init_guess_bypasses_table_lookup) {
    // The table lookup for the standard call uses a fallback (nearest entry),
    // so it never throws for valid nlap.  To demonstrate that the hint-based
    // overload truly bypasses the table, we use a non-tabulated range ratio by
    // constructing exponents/weights that are not in the standard table but are
    // a reasonable starting point.  We verify that the algorithm converges and
    // produces a positive errmax, i.e. it successfully ran the Remez loop.

    // Seed from a tabulated nearby run
    auto seed = minimax_cpppy::laplaceMinimax(5, 1.0, 100.0);

    // Use a larger interval (R=1e4) that is far from the seed's R=100 to
    // confirm the hint-based overload can still drive the algorithm to convergence.
    auto r = minimax_cpppy::laplaceMinimax(5, 1.0, 1e4, seed.expon, seed.weight);
    MINIMAX_REQUIRE(r.expon.size() == 5);
    MINIMAX_REQUIRE(r.weight.size() == 5);
    MINIMAX_REQUIRE(r.errmax > 0.0);
    const double err = max_approx_error(r, 1.0, 1e4);
    MINIMAX_REQUIRE(err < 1.0);  // very coarse: just confirm convergence, not divergence
}

// Edge: large ratio (1e12) — extrema are log-spaced over 12 decades
MINIMAX_TEST(nlap5_large_ratio_1e12) {
    auto r = minimax_cpppy::laplaceMinimax(5, 1.0, 1e12);
    MINIMAX_REQUIRE(r.expon.size() == 5);
    const double err = max_approx_error(r, 1.0, 1e12, 2000);
    std::cout << "    nlap=5 ratio=1e12 max error: " << err << "\n";
    MINIMAX_REQUIRE(err < 1e-1);
}

// Edge: small ratio near the minimum supported (10^0.1 ≈ 1.259)
MINIMAX_TEST(nlap5_small_ratio_near_min) {
    const double ratio = std::pow(10.0, 0.15);
    auto r = minimax_cpppy::laplaceMinimax(5, 1.0, ratio);
    MINIMAX_REQUIRE(r.expon.size() == 5);
    const double err = max_approx_error(r, 1.0, ratio, 1000);
    std::cout << "    nlap=5 ratio=10^0.15 max error: " << err << "\n";
    MINIMAX_REQUIRE(err < 1e-6);
}

// Guard triggers when interpolated extrema from the ext table fall outside [1, ratio].
// At ratio = 10^0.1 (the ext-table minimum) with large nlap the extrema are dense
// enough that Lagrange overshoot pushes some above the ratio.
MINIMAX_TEST(small_ratio_large_nlap_fallback) {
    const double ymin  = 1.0;
    const double ymax  = ymin * std::pow(10.0, 0.1);   // ratio ≈ 1.2589
    auto r = minimax_cpppy::laplaceMinimax(20, ymin, ymax);
    MINIMAX_REQUIRE(r.expon.size() == 20);
    MINIMAX_REQUIRE(r.weight.size() == 20);
    MINIMAX_REQUIRE(r.errmax > 0.0);
    const double err = max_approx_error(r, ymin, ymax, 1000);
    std::cout << "    nlap=20 ratio=10^0.1 max error: " << err << "\n";
    MINIMAX_REQUIRE(err < 1.0);
}

// Same regime, smaller nlap — confirms result is acceptable across a range.
MINIMAX_TEST(small_ratio_medium_nlap_fallback) {
    const double ymin  = 1.0;
    const double ymax  = ymin * std::pow(10.0, 0.1);
    auto r = minimax_cpppy::laplaceMinimax(10, ymin, ymax);
    MINIMAX_REQUIRE(r.expon.size() == 10);
    MINIMAX_REQUIRE(r.weight.size() == 10);
    MINIMAX_REQUIRE(r.errmax > 0.0);
    const double err = max_approx_error(r, ymin, ymax, 1000);
    std::cout << "    nlap=10 ratio=10^0.1 max error: " << err << "\n";
    MINIMAX_REQUIRE(err < 1.0);
}

int main() { MINIMAX_RUN_TESTS(); }
