// tests/test_quadrature.cpp
#include "test_helpers.hpp"
#include "core/quadrature.hpp"
#include <cmath>

using minimax_cpppy::detail::compositeGaussLegendre;
using minimax_cpppy::detail::QuadRule;

// Weights sum to the interval length.
MINIMAX_TEST(quad_weights_sum_to_length) {
    QuadRule r = compositeGaussLegendre(0.0, 3.0, 5);
    double s = 0.0;
    for (double w : r.w) s += w;
    MINIMAX_REQUIRE_CLOSE(s, 3.0, 1e-13);
    MINIMAX_REQUIRE(static_cast<int>(r.t.size()) == 16 * 5);
    MINIMAX_REQUIRE(r.t.size() == r.w.size());
}

// Integrates a smooth function exactly (to ~machine precision).
MINIMAX_TEST(quad_integrates_exp) {
    QuadRule r = compositeGaussLegendre(0.0, 2.0, 8);
    double integral = 0.0;
    for (size_t i = 0; i < r.t.size(); ++i) integral += r.w[i] * std::exp(-r.t[i]);
    // ∫_0^2 e^{-t} dt = 1 - e^{-2}
    MINIMAX_REQUIRE_CLOSE(integral, 1.0 - std::exp(-2.0), 1e-12);
}

int main() { MINIMAX_RUN_TESTS(); }
