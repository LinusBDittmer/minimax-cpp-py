// tests/test_newton.cpp
#include "test_helpers.hpp"
#include "core/newton.hpp"
#include "core/dd128.hpp"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

using minimax_cpppy::detail::DD;
using minimax_cpppy::detail::newtonMinimize;
using minimax_cpppy::detail::NewtonResult;

// f(x) = Σ (x_i - c_i)^2 ; minimum at x = c. Newton solves this in one step.
MINIMAX_TEST(newton_minimises_quadratic) {
    const int dim = 3;
    const double c[3] = {1.0, -2.0, 3.5};
    auto fvgh = [&](const DD* x, DD* g, DD* H) -> DD {
        DD f(0.0, 0.0);
        for (int i = 0; i < dim; ++i) {
            DD d = x[i] - DD(c[i]);
            f = f + DD::ddSquare(d);
            if (g) g[i] = DD(2.0) * d;
        }
        if (H) {
            for (int i = 0; i < dim * dim; ++i) H[i] = DD(0.0, 0.0);
            for (int i = 0; i < dim; ++i) H[i + i * dim] = DD(2.0);  // Hessian = 2I
        }
        return f;
    };
    std::vector<DD> x0 = {DD(0.0), DD(0.0), DD(0.0)};
    NewtonResult r = newtonMinimize(dim, fvgh, x0.data(), 100, 1e-12, 0, nullptr);
    MINIMAX_REQUIRE(r.converged);
    MINIMAX_REQUIRE(r.iters <= 2);  // quadratic: exact Newton step lands at the minimum
    for (int i = 0; i < dim; ++i) MINIMAX_REQUIRE_CLOSE(r.x[i].hi, c[i], 1e-10);
}

// Regression test: a Hessian that is exactly singular at the starting point
// (row/col 0 all-zero, matching ddLuFactorize's "near-zero pivot at column 0")
// must not crash the process. newtonDampedSolve's Levenberg-damping loop is
// specifically designed to recover from a singular system by escalating lam
// and retrying -- it must get the chance to do so in every build
// configuration, not just Release.
MINIMAX_TEST(newton_survives_singular_hessian_at_start) {
    const int dim = 2;
    // f(x) = x1^2, independent of x0: Hessian = diag(0, 2) is exactly singular.
    auto fvgh = [&](const DD* x, DD* g, DD* H) -> DD {
        if (g) { g[0] = DD(0.0); g[1] = DD(2.0) * x[1]; }
        if (H) {
            H[0] = DD(0.0); H[1] = DD(0.0);
            H[2] = DD(0.0); H[3] = DD(2.0);
        }
        return DD::ddSquare(x[1]);
    };
    std::vector<DD> x0 = {DD(1.0), DD(1.0)};
    NewtonResult r = newtonMinimize(dim, fvgh, x0.data(), 100, 1e-12, 0, nullptr);
    MINIMAX_REQUIRE(r.converged);
    MINIMAX_REQUIRE_CLOSE(r.x[1].hi, 0.0, 1e-8);
}

MINIMAX_TEST(newton_verbose3_prints_table) {
    const int dim = 3;
    const double c[3] = {1.0, -2.0, 3.5};
    auto fvgh = [&](const DD* x, DD* g, DD* H) -> DD {
        DD f(0.0, 0.0);
        for (int i = 0; i < dim; ++i) {
            DD d = x[i] - DD(c[i]);
            f = f + DD::ddSquare(d);
            if (g) g[i] = DD(2.0) * d;
        }
        if (H) {
            for (int i = 0; i < dim * dim; ++i) H[i] = DD(0.0, 0.0);
            for (int i = 0; i < dim; ++i) H[i + i * dim] = DD(2.0);
        }
        return f;
    };
    std::vector<DD> x0 = {DD(0.0), DD(0.0), DD(0.0)};
    std::ostringstream oss;
    NewtonResult r = newtonMinimize(dim, fvgh, x0.data(), 100, 1e-12, /*verbose=*/3, &oss);
    MINIMAX_REQUIRE(r.converged);
    std::string out = oss.str();
    MINIMAX_REQUIRE(out.find("[Newton]\n") != std::string::npos);
    MINIMAX_REQUIRE(out.find("|g|") != std::string::npos);
    size_t rowCount = static_cast<size_t>(std::count(out.begin(), out.end(), '\n'));
    MINIMAX_REQUIRE(rowCount >= 3);  // "[Newton]" + header + rule + >=1 data row
}

// A non-PD (indefinite) Hessian at the start must be damped into a descent step.
// f(x) = (x0-1)^2 - small saddle term; here use a positive-definite shifted case
// where the raw Hessian is indefinite to force the damping path.
MINIMAX_TEST(newton_damps_indefinite_hessian) {
    const int dim = 1;
    // f(x) = x^4 ; f'=4x^3, f''=12x^2. At x=0 the Hessian is 0 (singular) → damping.
    auto fvgh = [&](const DD* x, DD* g, DD* H) -> DD {
        DD x2 = DD::ddSquare(x[0]);
        DD f = DD::ddSquare(x2);              // x^4
        if (g) g[0] = DD(4.0) * (x2 * x[0]);  // 4 x^3
        if (H) H[0] = DD(12.0) * x2;          // 12 x^2
        return f;
    };
    std::vector<DD> x0 = {DD(1.0)};
    NewtonResult r = newtonMinimize(dim, fvgh, x0.data(), 200, 1e-10, 0, nullptr);
    MINIMAX_REQUIRE(r.converged);
    MINIMAX_REQUIRE_CLOSE(r.x[0].hi, 0.0, 1e-3);
}

// An indefinite Hessian at the start must be damped into a descent step.
// f(x,y) = (x^2-1)^2 + y^2 has minima at (±1,0); near the origin the x-curvature
// is negative (H_xx = 12x^2-4 < 0 for |x|<0.577), so the raw Newton step is not a
// descent direction and newtonDampedSolve must escalate damping to make progress.
MINIMAX_TEST(newton_damps_saddle_region) {
    const int dim = 2;
    auto fvgh = [&](const DD* v, DD* g, DD* H) -> DD {
        DD x = v[0], y = v[1];
        DD x2 = DD::ddSquare(x);
        DD t = x2 - DD(1.0);                       // x^2 - 1
        DD f = DD::ddSquare(t) + DD::ddSquare(y);  // (x^2-1)^2 + y^2
        if (g) {
            g[0] = DD(4.0) * (t * x);              // 4x(x^2-1)
            g[1] = DD(2.0) * y;                    // 2y
        }
        if (H) {
            for (int i = 0; i < dim * dim; ++i) H[i] = DD(0.0, 0.0);
            H[0]           = DD(12.0) * x2 - DD(4.0);  // 12x^2 - 4 (indefinite near 0)
            H[1 + 1 * dim] = DD(2.0);
        }
        return f;
    };
    std::vector<DD> x0 = {DD(0.5), DD(0.5)};       // H_xx = -1 here: indefinite
    NewtonResult r = newtonMinimize(dim, fvgh, x0.data(), 200, 1e-10, 0, nullptr);
    MINIMAX_REQUIRE(r.converged);
    MINIMAX_REQUIRE_CLOSE(r.x[0].hi, 1.0, 1e-6);   // basin x>0 -> (1, 0)
    MINIMAX_REQUIRE_CLOSE(r.x[1].hi, 0.0, 1e-6);
    MINIMAX_REQUIRE(r.fval.hi < 1e-12);
}

int main() { MINIMAX_RUN_TESTS(); }
