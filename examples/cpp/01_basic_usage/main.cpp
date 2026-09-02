/**
 * main.cpp — Introduction to the minimax_cpppy C++ library
 * ============================================================
 *
 * minimax_cpppy provides MINIMAX-optimal Laplace quadrature.  Given an
 * interval [ymin, ymax] and a number of quadrature points nlap, the library
 * returns exponents {a_k} and weights {w_k} such that
 *
 *   f(x) = sum_{k=0}^{nlap-1}  w_k * exp(-a_k * x)
 *
 * approximates  1/x  on [ymin, ymax] with the smallest possible maximum
 * absolute error (minimax / Chebyshev equioscillation criterion).
 *
 * Primary application: replace orbital-energy denominators  1/(D_ia + D_jb)
 * in MP2/CC theory with a product of two decaying exponentials, allowing the
 * four-index sum to factorise and reducing the computational scaling.
 *
 * Building this example
 * ---------------------
 * From the examples/cpp/01_basic_usage/ directory:
 *
 *   cmake -B build -S .
 *   cmake --build build
 *   ./build/basic_usage
 *
 * See CMakeLists.txt in this directory for how the library is linked.
 *
 * For a Debug build (enables parameter-validation assertions):
 *   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
 *   cmake --build build
 */

#include <algorithm>   // std::max
#include <cmath>       // std::exp, std::abs, std::pow
#include <cstdio>      // std::printf
#include <stdexcept>   // std::invalid_argument, std::runtime_error
#include <vector>

// The public header for minimax_cpppy.
// Everything lives in namespace minimax_cpppy.
#include "minimax_cpppy/minimax.hpp"
#include "minimax_cpppy/minimax_lp.hpp"   // laplaceLp — L_p-norm variant


// =============================================================================
// Helper utilities
// =============================================================================

static double evaluate_approx(
    const std::vector<double>& expon,
    const std::vector<double>& weight,
    double x)
{
    double result = 0.0;
    for (std::size_t k = 0; k < expon.size(); ++k)
        result += weight[k] * std::exp(-expon[k] * x);
    return result;
}

static double max_abs_error(
    const std::vector<double>& expon,
    const std::vector<double>& weight,
    double ymin, double ymax, int ngrid)
{
    double max_err = 0.0;
    for (int i = 0; i < ngrid; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(ngrid - 1);
        double x = ymin * std::pow(ymax / ymin, t);
        double err = std::abs(1.0 / x - evaluate_approx(expon, weight, x));
        max_err = std::max(max_err, err);
    }
    return max_err;
}


// =============================================================================
// main
// =============================================================================

int main()
{
    // =========================================================================
    // 1. A single call to laplaceMinimax
    // =========================================================================

    const int    nlap = 7;
    const double ymin = 1.0;
    const double ymax = 1000.0;

    // laplaceMinimax is the one public function in the library.
    //
    // Parameters
    //   nlap         — number of quadrature points, 1 <= nlap <= 30
    //   ymin, ymax   — interval boundaries, 0 < ymin < ymax
    //
    // Returns a MinimaxResult struct with three members:
    //   .expon   — std::vector<double> of length nlap   (exponents a_k)
    //   .weight  — std::vector<double> of length nlap   (weights w_k)
    //   .errmax  — double, guaranteed max |1/x - f(x)| on [ymin, ymax]
    //
    // Throws std::invalid_argument (Debug) / std::runtime_error (Release)
    // if nlap is outside [1, 30].
    minimax_cpppy::MinimaxResult r = minimax_cpppy::laplaceMinimax(nlap, ymin, ymax);

    std::printf("=============================================================\n");
    std::printf("laplaceMinimax(nlap=%d, ymin=%.1f, ymax=%.1f)\n", nlap, ymin, ymax);
    std::printf("=============================================================\n");
    std::printf("  Guaranteed max error (errmax) = %.3e\n\n", r.errmax);

    std::printf("  %3s   %-22s   %-22s\n", "k", "exponent a_k", "weight w_k");
    std::printf("  %3s   %-22s   %-22s\n", "---",
                "----------------------", "----------------------");
    for (int k = 0; k < nlap; ++k)
        std::printf("  %3d   %22.14f   %22.14f\n", k, r.expon[k], r.weight[k]);
    std::printf("\n");


    // =========================================================================
    // 2. Verify the approximation manually
    // =========================================================================

    const int ngrid   = 500;
    double    man_err = max_abs_error(r.expon, r.weight, ymin, ymax, ngrid);

    std::printf("=============================================================\n");
    std::printf("Verification on %d log-spaced points in [%.1f, %.1f]\n",
                ngrid, ymin, ymax);
    std::printf("=============================================================\n");
    std::printf("  Max absolute error (manual) = %.3e\n", man_err);
    std::printf("  Guaranteed max error (errmax) = %.3e\n\n", r.errmax);


    // =========================================================================
    // 3. Spot check at a specific x value
    // =========================================================================

    const double x_check  = 42.0;
    const double f_approx = evaluate_approx(r.expon, r.weight, x_check);
    const double f_exact  = 1.0 / x_check;
    const double abs_err  = std::abs(f_exact - f_approx);

    std::printf("=============================================================\n");
    std::printf("Spot check at x = %.1f\n", x_check);
    std::printf("=============================================================\n");
    std::printf("  Exact   1/x  = %.14f\n", f_exact);
    std::printf("  Approx  f(x) = %.14f\n", f_approx);
    std::printf("  |error|      = %.3e   (errmax = %.3e)\n\n", abs_err, r.errmax);


    // =========================================================================
    // 4. Convergence: fewer vs more quadrature points
    // =========================================================================

    std::printf("=============================================================\n");
    std::printf("Convergence: max error on [%.1f, %.1f] vs nlap\n", ymin, ymax);
    std::printf("=============================================================\n");
    std::printf("  %5s   %-22s   %-22s\n",
                "nlap", "errmax (library)", "error (manual)");
    std::printf("  %5s   %-22s   %-22s\n",
                "-----", "----------------------", "----------------------");

    for (int n : {2, 3, 5, 7, 10, 15}) {
        try {
            minimax_cpppy::MinimaxResult res =
                minimax_cpppy::laplaceMinimax(n, ymin, ymax);
            double me = max_abs_error(res.expon, res.weight, ymin, ymax, 1000);
            std::printf("  %5d   %22.6e   %22.6e\n", n, res.errmax, me);
        } catch (const std::runtime_error& exc) {
            std::printf("  %5d   (RuntimeError: %s)\n", n, exc.what());
        }
    }
    std::printf("\n");


    // =========================================================================
    // 5. Quantum-chemistry motivating example: orbital-energy denominators
    // =========================================================================
    //
    // In MP2 / coupled-cluster theory, the energy denominator is
    //
    //   1 / (D_ia + D_jb)   where   D_ia = eps_a - eps_i > 0
    //
    // The Laplace transform replaces this with
    //
    //   sum_k  w_k * exp(-t_k * D_ia) * exp(-t_k * D_jb)
    //
    // which factorises the four-index (i,j,a,b) sum into two independent
    // two-index sums, reducing the leading computational cost.

    std::printf("=============================================================\n");
    std::printf("Example: orbital-energy denominators (mock data)\n");
    std::printf("=============================================================\n");

    const std::vector<double> D_values = {
        0.30, 0.55, 0.82, 1.40, 2.10, 3.80, 6.50, 11.0, 18.0
    };

    const double ymin_chem = 2.0 * D_values.front();
    const double ymax_chem = 2.0 * D_values.back();

    const int nlap_chem = 5;
    minimax_cpppy::MinimaxResult r_chem =
        minimax_cpppy::laplaceMinimax(nlap_chem, ymin_chem, ymax_chem);

    std::printf("  Denominator range: [%.2f, %.2f] Eh  |  nlap=%d\n",
                ymin_chem, ymax_chem, nlap_chem);
    std::printf("  Guaranteed errmax = %.3e\n\n", r_chem.errmax);

    std::printf("  %10s   %14s   %14s   %12s\n",
                "D_ia (Eh)", "1/D_ia (exact)", "1/D_ia (LT)", "|error|");
    std::printf("  %10s   %14s   %14s   %12s\n",
                "----------", "--------------", "--------------", "------------");

    for (double D : D_values) {
        double approx_D = evaluate_approx(r_chem.expon, r_chem.weight, D);
        double exact_D  = 1.0 / D;
        double err_D    = std::abs(exact_D - approx_D);
        std::printf("  %10.4f   %14.8f   %14.8f   %12.3e\n",
                    D, exact_D, approx_D, err_D);
    }
    std::printf("\n");


    // =========================================================================
    // 6. L_p-norm-optimal quadrature (laplaceLp)
    // =========================================================================
    //
    // laplaceMinimax minimises the L_inf (worst-case) error — equioscillation.
    // laplaceLp instead minimises the L_p norm of the error for a real
    // exponent normP >= 1:
    //
    //   ||eta||_p = (1/ymin) * ( integral |eta(t)|^p dt )^(1/p)
    //
    // Small normP spreads the error out (lower mean error, higher peak); as
    // normP grows the solution approaches the pure minimax (L_inf) result.
    // Note the returned .errmax field here holds the L_p NORM, not the L_inf
    // error — so we also measure the true peak error manually for comparison.

    std::printf("=============================================================\n");
    std::printf("L_p-norm-optimal quadrature (nlap=%d, [%.1f, %.1f])\n",
                nlap, ymin, ymax);
    std::printf("=============================================================\n");
    std::printf("  %6s   %-18s   %-18s\n",
                "norm_p", "L_p norm (.errmax)", "peak L_inf error");
    std::printf("  %6s   %-18s   %-18s\n",
                "------", "------------------", "------------------");

    for (double normP : {2.0, 4.0, 8.0, 16.0}) {
        try {
            minimax_cpppy::MinimaxResult rn =
                minimax_cpppy::laplaceLp(nlap, ymin, ymax, normP);
            double peak = max_abs_error(rn.expon, rn.weight, ymin, ymax, 1000);
            std::printf("  %6.1f   %18.6e   %18.6e\n", normP, rn.errmax, peak);
        } catch (const std::runtime_error& exc) {
            std::printf("  %6.1f   (RuntimeError: %s)\n", normP, exc.what());
        }
    }
    std::printf("  (compare: pure minimax L_inf errmax = %.6e)\n\n", r.errmax);


    // =========================================================================
    // 7. Error handling
    // =========================================================================

    std::printf("=============================================================\n");
    std::printf("Error handling\n");
    std::printf("=============================================================\n");

    // nlap outside [1, 30] — always throws invalid_argument
    try {
        minimax_cpppy::laplaceMinimax(31, 1.0, 10.0);
        std::printf("  nlap=31 → (no exception thrown)\n");
    } catch (const std::invalid_argument& exc) {
        std::printf("  nlap=31 → invalid_argument: %s\n", exc.what());
    }

    // inverted interval — always throws invalid_argument
    try {
        minimax_cpppy::laplaceMinimax(5, 10.0, 1.0);
        std::printf("  ymax<ymin → (no exception thrown)\n");
    } catch (const std::invalid_argument& exc) {
        std::printf("  ymax<ymin → invalid_argument: %s\n", exc.what());
    }

    std::printf("\nDone.  See 02_denominator_density/ for DenominatorDensity usage.\n");
    return 0;
}
