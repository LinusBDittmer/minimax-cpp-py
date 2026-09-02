#ifndef MINIMAX_CPPPY_NEWTON_HPP
#define MINIMAX_CPPPY_NEWTON_HPP

#include <cmath>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <vector>
#include "dd128.hpp"
#include "log_pretty.hpp"

namespace minimax_cpppy {
namespace detail {

struct NewtonResult {
    std::vector<DD> x;
    DD     fval{0.0, 0.0};
    double gradNormInf = 0.0;
    int    iters = 0;
    bool   converged = false;
};

inline double newtonInfNorm(int dim, const DD* v) {
    double m = 0.0;
    for (int i = 0; i < dim; ++i) { double a = std::abs(v[i].hi); if (a > m) m = a; }
    return m;
}

// Solve (H + lam I) delta = -g for a descent direction, escalating the
// Levenberg damping lam until the step is finite and gᵀdelta < 0. As lam grows
// the system tends to lam·delta = -g (scaled steepest descent), so a descent
// direction is always eventually found; -g is the final fallback.
inline void newtonDampedSolve(int dim, const DD* H, const DD* g, DD* delta) {
    double hinf = 0.0;  // scale for the initial nonzero damping
    for (int i = 0; i < dim; ++i) {
        double d = std::abs(H[i + i * dim].hi);
        if (d > hinf) hinf = d;
    }
    if (hinf < 1.0) hinf = 1.0;

    // A singular damped system at the current lam is an expected, recoverable
    // case, not a bug: Release builds see it as inf/nan from ddLuFactorize
    // (caught by the finiteness check below); Debug builds additionally throw
    // std::runtime_error on a near-zero pivot, which is caught here and treated
    // identically -- either way, the loop below escalates lam and retries.
    std::vector<DD> M(static_cast<size_t>(dim) * dim);
    std::vector<int> piv(dim);
    double lam = 0.0;
    for (int attempt = 0; attempt < 40; ++attempt) {
        M.assign(H, H + static_cast<size_t>(dim) * dim);
        for (int i = 0; i < dim; ++i) M[i + i * dim] = M[i + i * dim] + DD(lam);
        std::vector<DD> rhs(dim);
        for (int i = 0; i < dim; ++i) rhs[i] = -g[i];
        bool ok = true;
        try {
            ddLuFactorize(M.data(), dim, piv.data());
            ddLuSolve(M.data(), dim, piv.data(), rhs.data());
        } catch (const std::exception&) {
            ok = false;
        }
        DD slope(0.0, 0.0);
        for (int i = 0; ok && i < dim; ++i) {
            if (!std::isfinite(rhs[i].hi)) { ok = false; break; }
            slope = slope + g[i] * rhs[i];
        }
        if (ok && slope.hi < 0.0) {
            for (int i = 0; i < dim; ++i) delta[i] = rhs[i];
            return;
        }
        lam = (lam == 0.0) ? 1e-12 * hinf : lam * 10.0;
    }
    for (int i = 0; i < dim; ++i) delta[i] = -g[i];  // steepest-descent fallback
}

// Damped Newton minimiser. fValGradHess(x, grad, hess) returns f(x); when grad
// (length dim) is non-null it is filled, and when hess (dim*dim column-major) is
// non-null it is filled. Converged when ‖grad‖_∞ < tol·max(1, ‖grad_0‖_∞).
inline NewtonResult newtonMinimize(
    int dim, const std::function<DD(const DD*, DD*, DD*)>& fValGradHess,
    const DD* x0, int maxIter, double tol, int verbose, std::ostream* os) {
    NewtonResult R;
    R.x.assign(x0, x0 + dim);
    std::vector<DD> g(dim), delta(dim), xNew(dim);
    std::vector<DD> H(static_cast<size_t>(dim) * dim);

    DD f = fValGradHess(R.x.data(), g.data(), H.data());
    const double g0 = newtonInfNorm(dim, g.data());
    const double stop = tol * (g0 > 1.0 ? g0 : 1.0);

    if (verbose >= 3 && os) {
        *os << "[Newton]\n"
            << fmtHeaderCell("iter", 4) << "  " << fmtHeaderCell("f", 12)
            << "  " << fmtHeaderCell("|g|", 12) << "\n"
            << ruleCell(4) << "  " << ruleCell(12) << "  " << ruleCell(12) << "\n";
    }
    R.iters = maxIter;  // overwritten by a break; correct value if the loop exhausts
    for (int iter = 0; iter < maxIter; ++iter) {
        double gnorm = newtonInfNorm(dim, g.data());
        if (verbose >= 3 && os) {
            *os << fmtCellInt(iter, 4) << "  " << fmtCell(f.hi, 12, 4)
                << "  " << fmtCell(gnorm, 12, 4) << "\n";
        }
        if (gnorm < stop) { R.converged = true; R.iters = iter; break; }

        newtonDampedSolve(dim, H.data(), g.data(), delta.data());

        DD slope(0.0, 0.0);
        for (int i = 0; i < dim; ++i) slope = slope + g[i] * delta[i];

        // Armijo backtracking line search.
        double alpha = 1.0;
        const double c1 = 1e-4;
        DD fNew;
        bool stepOk = false;
        for (int ls = 0; ls < 50; ++ls) {
            for (int i = 0; i < dim; ++i) xNew[i] = R.x[i] + alpha * delta[i];
            fNew = fValGradHess(xNew.data(), nullptr, nullptr);
            if (std::isfinite(fNew.hi) &&
                fNew.hi <= f.hi + c1 * alpha * slope.hi) { stepOk = true; break; }
            alpha *= 0.5;
        }
        if (!stepOk) { R.iters = iter; break; }  // line search failed → not converged

        R.x = xNew;
        f = fValGradHess(R.x.data(), g.data(), H.data());
    }

    R.fval = f;
    R.gradNormInf = newtonInfNorm(dim, g.data());
    return R;
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_NEWTON_HPP
