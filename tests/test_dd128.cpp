#include "test_helpers.hpp"
#include "core/dd128.hpp"
#include <cmath>

using DD = minimax_cpppy::detail::DD;

// log(1) = 0 exactly
MINIMAX_TEST(ddLog_of_one_is_zero) {
    DD result = DD::ddLog(DD(1.0, 0.0));
    MINIMAX_REQUIRE(std::abs(result.hi) < 1e-30);
    MINIMAX_REQUIRE(std::abs(result.lo) < 1e-30);
}

// log(e) = 1; std::exp(1.0) is the closest double to e
MINIMAX_TEST(ddLog_of_e_is_one) {
    DD result = DD::ddLog(DD(std::exp(1.0), 0.0));
    MINIMAX_REQUIRE(std::abs(result.hi - 1.0) < 1e-14);
}

// log(exp(x)) ≈ x for the range of t values maehlySolver uses
MINIMAX_TEST(ddLog_is_inverse_of_ddExp_in_solver_range) {
    for (double x : {0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 27.6}) {  // 27.6 ≈ log(1e12)
        DD ex  = DD::ddExp(DD(x, 0.0));
        DD lex = DD::ddLog(ex);
        // Should recover x to better than 1e-14
        MINIMAX_REQUIRE(std::abs(lex.hi - x) < 1e-14);
    }
}

// For exact double inputs, hi part should match std::log to ~1 ulp
MINIMAX_TEST(ddLog_hi_matches_std_log) {
    for (double x : {2.0, std::exp(1.0), 10.0, 100.0, 1e6, 1e12}) {
        DD result  = DD::ddLog(DD(x, 0.0));
        double ref = std::log(x);
        // hi should be within 2 ulps of std::log
        MINIMAX_REQUIRE(std::abs(result.hi - ref) <= 2.0 * std::abs(ref) * 2.22e-16);
    }
}

// lo correction is non-trivial for non-exact inputs
MINIMAX_TEST(ddLog_lo_correction_improves_accuracy) {
    // ddExp gives a non-trivial lo part; ddLog should recover the value better than just std::log(hi)
    DD ex  = DD::ddExp(DD(5.0, 0.0));           // hi+lo ≈ e^5
    DD lex = DD::ddLog(ex);
    double err_with_lo    = std::abs(lex.hi + lex.lo - 5.0);
    double err_without_lo = std::abs(lex.hi        - 5.0);
    // Using lo should not make things worse (and usually better)
    MINIMAX_REQUIRE(err_with_lo <= err_without_lo + 1e-30);
}

MINIMAX_TEST(ddExp_just_below_overflow_guard_matches_std_exp) {
    DD r = DD::ddExp(DD(700.0, 0.0));
    MINIMAX_REQUIRE(std::isfinite(r.hi));
    MINIMAX_REQUIRE_CLOSE(r.hi, std::exp(700.0), 1e-10);
}

MINIMAX_TEST(ddExp_above_overflow_guard_returns_infinity_not_hang) {
    for (double x : {701.0, 1e19, 1e20, 1e100, 1e167}) {
        DD r = DD::ddExp(DD(x, 0.0));
        MINIMAX_REQUIRE(!std::isfinite(r.hi));
        MINIMAX_REQUIRE(r.hi > 0.0);
    }
}

MINIMAX_TEST(ddExp_negative_of_large_positive_still_underflows_to_zero) {
    // Existing XMIN path must remain unaffected by the new XMAX guard.
    DD r = DD::ddExp(DD(-1e19, 0.0));
    MINIMAX_REQUIRE(r.hi == 0.0);
}

MINIMAX_TEST(ddExp_mid_range_previously_broken_now_accurate) {
    for (double x : {69.5, 80.0, 100.0, 200.0, 500.0}) {
        DD r = DD::ddExp(DD(x, 0.0));
        MINIMAX_REQUIRE(std::isfinite(r.hi));
        MINIMAX_REQUIRE_CLOSE(r.hi, std::exp(x), 1e-10);
    }
}

#ifdef MINIMAX_CPPPY_DEBUG_MODE__
MINIMAX_TEST(ddLog_zero_throws_in_debug) {
    MINIMAX_REQUIRE_THROW_TYPE(DD::ddLog(DD(0.0, 0.0)), std::invalid_argument);
}

MINIMAX_TEST(ddLog_negative_throws_in_debug) {
    MINIMAX_REQUIRE_THROW_TYPE(DD::ddLog(DD(-1.0, 0.0)), std::invalid_argument);
}
#endif

int main() { MINIMAX_RUN_TESTS(); }
