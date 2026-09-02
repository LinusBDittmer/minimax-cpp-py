/**
 * main.cpp — Biased vs standard minimax quadrature
 * =================================================
 *
 * Standard laplaceMinimax minimises the uniform L-infinity error
 *
 *   max_{x in [ymin,ymax]}  |1/x - f(x)|
 *
 * treating all denominator values equally.
 *
 * biasedLaplace instead minimises the density-weighted error
 *
 *   max_{x in [ymin,ymax]}  w(t(x)) * |1/x - f(x)|
 *
 * where  t(x) = ln(x / delta_min)  and  w(t) = p_t(t)  is the
 * DenominatorDensity evaluated at t.  This relaxes the error
 * guarantee in sparsely populated regions (large denominators)
 * and tightens it where the density is high (near the HOMO-LUMO
 * gap), typically improving the physical correlation-energy error
 * for the same number of quadrature points.
 *
 * Building
 * --------
 * From the examples/cpp/03_biased_vs_unbiased/ directory:
 *
 *   cmake -B build -S .
 *   cmake --build build
 *   ./build/biased_vs_unbiased
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "minimax_cpppy/minimax.hpp"
#include "minimax_cpppy/biasing.hpp"


// =============================================================================
// H2O RHF/cc-pVDZ orbital energies (Hartree)
// =============================================================================

static constexpr double H2O_OCC[] = {
    -20.550438, -1.336658, -0.699262, -0.566562, -0.493142
};
static constexpr double H2O_VIRT[] = {
    0.185559, 0.256244, 0.789271, 0.854276, 1.163512,
    1.200385, 1.253306, 1.444602, 1.476247, 1.674666,
    1.867313, 1.934850, 2.452768, 2.490253, 3.285619,
    3.338865, 3.510478, 3.865845, 4.147450
};
static constexpr int N_OCC  = 5;
static constexpr int N_VIRT = 19;


// =============================================================================
// Helper: evaluate f(x) = sum_k w_k * exp(-a_k * x)
// =============================================================================

static double eval_approx(
    const std::vector<double>& expon,
    const std::vector<double>& weight,
    double x)
{
    double s = 0.0;
    for (std::size_t k = 0; k < expon.size(); ++k)
        s += weight[k] * std::exp(-expon[k] * x);
    return s;
}


int main()
{
    const int nlap = 6;

    // =========================================================================
    // 1. Build density and run both optimisations
    // =========================================================================

    minimax_cpppy::DenominatorDensity d(
        H2O_OCC,  N_OCC,
        H2O_VIRT, N_VIRT,
        1.0 /* bandwidth */);

    const double ymin = d.deltaMin();
    const double ymax = d.deltaMax();

    // Standard minimax: equal weight on all x in [ymin, ymax]
    minimax_cpppy::MinimaxResult r_u =
        minimax_cpppy::laplaceMinimax(nlap, ymin, ymax);

    // Biased minimax: error weighted by p_t(t(x)); builds its own density
    // internally from the same (occ, virt, bandwidth) inputs.
    minimax_cpppy::MinimaxResult r_b =
        minimax_cpppy::biasedLaplace(
            nlap, ymin, ymax,
            H2O_OCC,  N_OCC,
            H2O_VIRT, N_VIRT,
            1.0 /* bandwidth */);

    std::printf("=============================================================\n");
    std::printf("H2O denominator interval:  [%.4f, %.4f] Eh  (nlap=%d)\n",
                ymin, ymax, nlap);
    std::printf("=============================================================\n\n");


    // =========================================================================
    // 2. Quadrature nodes side-by-side
    // =========================================================================

    std::printf("Exponents (a_k):\n");
    std::printf("  %3s   %20s   %20s   %12s\n",
                "k", "unbiased", "biased", "shift");
    std::printf("  %3s   %20s   %20s   %12s\n",
                "---", "--------------------", "--------------------",
                "------------");
    for (int k = 0; k < nlap; ++k) {
        double shift = r_b.expon[k] - r_u.expon[k];
        std::printf("  %3d   %20.12f   %20.12f   %+12.4e\n",
                    k, r_u.expon[k], r_b.expon[k], shift);
    }
    std::printf("\n");

    std::printf("Weights (w_k):\n");
    std::printf("  %3s   %20s   %20s   %12s\n",
                "k", "unbiased", "biased", "shift");
    std::printf("  %3s   %20s   %20s   %12s\n",
                "---", "--------------------", "--------------------",
                "------------");
    for (int k = 0; k < nlap; ++k) {
        double shift = r_b.weight[k] - r_u.weight[k];
        std::printf("  %3d   %20.12f   %20.12f   %+12.4e\n",
                    k, r_u.weight[k], r_b.weight[k], shift);
    }
    std::printf("\n");


    // =========================================================================
    // 3. Pointwise error at representative x values
    // =========================================================================
    //
    // Sample at t_norm values spread across [0, 1] where t_norm = t/ln(ratio).
    // Small t_norm → small denominator (high-density region).
    // Large t_norm → large denominator (sparse region).

    const double t_max    = std::log(d.ratio());
    const double t_norms[] = {0.02, 0.15, 0.30, 0.50, 0.65, 0.80, 0.98};
    const int    n_samples = static_cast<int>(sizeof(t_norms) / sizeof(*t_norms));

    std::printf("Pointwise approximation error |1/x - f(x)|:\n");
    std::printf("  t_norm = t / ln(ratio):  0 = small denominator, 1 = large\n");
    std::printf("\n");
    std::printf("  %7s   %10s   %14s   %14s   %7s\n",
                "t_norm", "x (Eh)", "err_unbiased", "err_biased", "  ratio");
    std::printf("  %7s   %10s   %14s   %14s   %7s\n",
                "-------", "----------", "--------------",
                "--------------", "-------");

    for (int i = 0; i < n_samples; ++i) {
        double t  = t_norms[i] * t_max;
        double x  = ymin * std::exp(t);
        double eu = std::abs(1.0 / x - eval_approx(r_u.expon, r_u.weight, x));
        double eb = std::abs(1.0 / x - eval_approx(r_b.expon, r_b.weight, x));
        char cmp  = (eb < eu) ? '<' : '>';
        std::printf("  %7.2f   %10.4f   %14.3e   %14.3e   %c 1 (%.3f)\n",
                    t_norms[i], x, eu, eb, cmp, (eu > 0.0 ? eb / eu : 0.0));
    }
    std::printf("\n");


    // =========================================================================
    // 4. errmax and global summary
    // =========================================================================
    //
    // r_u.errmax is the unweighted L-infinity maximum.
    // r_b.errmax is the density-weighted maximum — a different criterion,
    // not directly comparable to r_u.errmax.
    //
    // Compute unweighted max error for both to compare on the same scale.

    const int    n_check = 2000;
    double eu_max = 0.0, eb_max = 0.0;
    int    n_biased_better = 0;

    for (int i = 0; i < n_check; ++i) {
        double t = t_max * i / (n_check - 1);
        double x = ymin * std::exp(t);
        double eu = std::abs(1.0 / x - eval_approx(r_u.expon, r_u.weight, x));
        double eb = std::abs(1.0 / x - eval_approx(r_b.expon, r_b.weight, x));
        eu_max = std::max(eu_max, eu);
        eb_max = std::max(eb_max, eb);
        if (eb < eu) ++n_biased_better;
    }

    std::printf("Global error summary (unweighted L-infinity):\n");
    std::printf("  Unbiased: errmax (library) = %.3e\n", r_u.errmax);
    std::printf("            errmax (manual)  = %.3e\n", eu_max);
    std::printf("\n");
    std::printf("  Biased:   errmax (library) = %.3e  [density-weighted criterion]\n",
                r_b.errmax);
    std::printf("            errmax (manual)  = %.3e  [unweighted, for comparison]\n",
                eb_max);
    std::printf("\n");
    std::printf("  Biased has smaller unweighted error at %d / %d (%.0f%%) sampled x.\n",
                n_biased_better, n_check,
                100.0 * n_biased_better / n_check);
    std::printf("  The region where biased wins corresponds to the density peak\n");
    std::printf("  (t*/t_max ~ 0.81 for H2O) — not necessarily small denominators.\n");
    std::printf("\nDone.\n");

    return 0;
}
