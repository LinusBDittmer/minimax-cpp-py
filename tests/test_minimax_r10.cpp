#include "test_helpers.hpp"
#include "minimax_cpppy/minimax.hpp"
#include <cmath>
#include <iostream>

static constexpr double YMIN = 1.0;
static constexpr double YMAX = 10.0;

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

static void run_convergence_test(int nlap, int verbose = 0)
{
    auto r = minimax_cpppy::laplaceMinimax(nlap, YMIN, YMAX, verbose, std::cout);
    MINIMAX_REQUIRE(static_cast<int>(r.expon.size()) == nlap);
    MINIMAX_REQUIRE(static_cast<int>(r.weight.size()) == nlap);
    MINIMAX_REQUIRE(r.errmax > 0.0);
    const double err = max_approx_error(r, YMIN, YMAX);
    std::cout << "    nlap=" << nlap << " errmax=" << r.errmax
              << " approx_err=" << err << "\n";
    MINIMAX_REQUIRE(err < 0.5);
}

#define MAKE_NLAP_TEST(n) \
    MINIMAX_TEST(nlap_##n##_ymin1_ymax10) { run_convergence_test(n); }
MAKE_NLAP_TEST(1)
MAKE_NLAP_TEST(2)
MAKE_NLAP_TEST(3)
MAKE_NLAP_TEST(4)
MAKE_NLAP_TEST(5)
MAKE_NLAP_TEST(6)
MAKE_NLAP_TEST(7)
MAKE_NLAP_TEST(8)
MAKE_NLAP_TEST(9)
MAKE_NLAP_TEST(10)
MAKE_NLAP_TEST(11)
MAKE_NLAP_TEST(12)
MAKE_NLAP_TEST(13)
MAKE_NLAP_TEST(14)
MAKE_NLAP_TEST(15)
MAKE_NLAP_TEST(16)
MAKE_NLAP_TEST(17)
MAKE_NLAP_TEST(18)
MAKE_NLAP_TEST(19)
MAKE_NLAP_TEST(20)

int main() { MINIMAX_RUN_TESTS(); }
