#include "test_helpers.hpp"
#include "core/algorithm.hpp"
#include <cmath>

using DD = minimax_cpppy::detail::DD;

static void compute_func_grad(DD& func, DD& grad, DD x,
                               const DD* exp_arr, const DD* wgt_arr, int nlap)
{
    DD xSq = DD::ddSquare(x);
    func   = DD(1.0) / xSq;
    grad   = DD(-2.0) / (xSq * x);
    for (int k = 0; k < nlap; ++k) {
        DD et  = DD::ddExp(-(exp_arr[k] * x));
        DD wet = exp_arr[k] * wgt_arr[k] * et;
        func  -= wet;
        grad  += exp_arr[k] * wet;
    }
}

// nlap=1, t=0 (x=1), a=1, w=1:
//   u = 1, exp(-1) ≈ 0.36788
//   h'(0)  = -1 + exp(-1) ≈ -0.63212
//   h''(0) =  1 + 0 = 1.0
MINIMAX_TEST(evalLogErrorDerivatives_nlap1_t0_analytic) {
    DD exp_arr[1] = { DD(1.0, 0.0) };
    DD wgt_arr[1] = { DD(1.0, 0.0) };
    DD hprime, h2prime;
    minimax_cpppy::detail::evalLogErrorDerivatives(
        hprime, h2prime, DD(0.0, 0.0), exp_arr, wgt_arr, 1);
    const double expected_hprime  = -1.0 + std::exp(-1.0);
    const double expected_h2prime =  1.0;
    MINIMAX_REQUIRE(std::abs(hprime.hi  - expected_hprime)  < 1e-14);
    MINIMAX_REQUIRE(std::abs(h2prime.hi - expected_h2prime) < 1e-14);
}

MINIMAX_TEST(evalLogErrorDerivatives_identity_nlap2) {
    DD exp_arr[2] = { DD(0.5, 0.0), DD(2.0, 0.0) };
    DD wgt_arr[2] = { DD(1.5, 0.0), DD(0.3, 0.0) };
    DD t = DD::ddLog(DD(2.0, 0.0));
    DD x = DD::ddExp(t);

    DD hprime, h2prime;
    minimax_cpppy::detail::evalLogErrorDerivatives(
        hprime, h2prime, t, exp_arr, wgt_arr, 2);

    DD func, grad;
    compute_func_grad(func, grad, x, exp_arr, wgt_arr, 2);

    double expected_hprime  = -x.hi * func.hi;
    double expected_h2prime = expected_hprime - x.hi * x.hi * grad.hi;

    MINIMAX_REQUIRE(std::abs(hprime.hi  - expected_hprime)  < 1e-12);
    MINIMAX_REQUIRE(std::abs(h2prime.hi - expected_h2prime) < 1e-12);
}

MINIMAX_TEST(evalLogErrorDerivatives_identity_nlap3_t_log100) {
    DD exp_arr[3] = { DD(0.01, 0.0), DD(0.3, 0.0), DD(5.0, 0.0) };
    DD wgt_arr[3] = { DD(0.5,  0.0), DD(1.2, 0.0), DD(0.1, 0.0) };
    DD t = DD::ddLog(DD(100.0, 0.0));
    DD x = DD::ddExp(t);

    DD hprime, h2prime;
    minimax_cpppy::detail::evalLogErrorDerivatives(
        hprime, h2prime, t, exp_arr, wgt_arr, 3);

    DD func, grad;
    compute_func_grad(func, grad, x, exp_arr, wgt_arr, 3);

    double expected_hprime  = -x.hi * func.hi;
    double expected_h2prime = expected_hprime - x.hi * x.hi * grad.hi;

    MINIMAX_REQUIRE(std::abs(hprime.hi  - expected_hprime)  < 1e-10);
    MINIMAX_REQUIRE(std::abs(h2prime.hi - expected_h2prime) < 1e-10);
}

MINIMAX_TEST(evalLogErrorDerivatives_underflow_safe_large_u) {
    DD exp_arr[1] = { DD(1e6, 0.0) };
    DD wgt_arr[1] = { DD(1.0, 0.0) };
    DD t = DD(std::log(1e12), 0.0);
    DD hprime, h2prime;
    minimax_cpppy::detail::evalLogErrorDerivatives(
        hprime, h2prime, t, exp_arr, wgt_arr, 1);
    MINIMAX_REQUIRE(std::isfinite(hprime.hi));
    MINIMAX_REQUIRE(std::isfinite(h2prime.hi));
    const double invx = 1.0 / 1e12;
    MINIMAX_REQUIRE(std::abs(hprime.hi  - (-invx)) < 1e-20);
    MINIMAX_REQUIRE(std::abs(h2prime.hi - ( invx)) < 1e-20);
}

MINIMAX_TEST(evalLogErrorDerivatives_underflow_safe_large_a) {
    DD exp_arr[1] = { DD(1000.0, 0.0) };
    DD wgt_arr[1] = { DD(1.0,   0.0) };
    DD hprime, h2prime;
    minimax_cpppy::detail::evalLogErrorDerivatives(
        hprime, h2prime, DD(0.0, 0.0), exp_arr, wgt_arr, 1);
    MINIMAX_REQUIRE(std::isfinite(hprime.hi));
    MINIMAX_REQUIRE(std::isfinite(h2prime.hi));
    MINIMAX_REQUIRE(std::abs(hprime.hi  - (-1.0)) < 1e-14);
    MINIMAX_REQUIRE(std::abs(h2prime.hi - ( 1.0)) < 1e-14);
}

#ifdef MINIMAX_CPPPY_DEBUG_MODE__
MINIMAX_TEST(evalLogErrorDerivatives_null_exponents_throws) {
    DD wgt_arr[1] = { DD(1.0, 0.0) };
    DD hprime, h2prime;
    MINIMAX_REQUIRE_THROW_TYPE(
        minimax_cpppy::detail::evalLogErrorDerivatives(
            hprime, h2prime, DD(0.0), nullptr, wgt_arr, 1),
        std::invalid_argument);
}

MINIMAX_TEST(evalLogErrorDerivatives_nonfinite_t_throws) {
    DD exp_arr[1] = { DD(1.0, 0.0) };
    DD wgt_arr[1] = { DD(1.0, 0.0) };
    DD hprime, h2prime;
    double inf = std::numeric_limits<double>::infinity();
    MINIMAX_REQUIRE_THROW_TYPE(
        minimax_cpppy::detail::evalLogErrorDerivatives(
            hprime, h2prime, DD(inf), exp_arr, wgt_arr, 1),
        std::invalid_argument);
}
#endif

int main() { MINIMAX_RUN_TESTS(); }
