/**
 * main.cpp — DenominatorDensity: pairwise denominator distribution
 * =================================================================
 *
 * In MP2/CC theory the orbital-energy denominator for pair (ia, jb) is
 *
 *   Delta_{ia,jb} = D_{ia} + D_{jb}    where   D_{ia} = eps_a - eps_i > 0
 *
 * Standard minimax quadrature treats all denominator values equally.  But the
 * denominators are NOT uniformly distributed — near the HOMO-LUMO gap (small
 * D_{ia}) there are many pairs; at large D_{ia} there are few.
 *
 * DenominatorDensity computes the probability density p_t(t) of the
 * log-denominator  t = ln(Delta / delta_min)  via FFT-based KDE, then fits a
 * quintic Hermite spline for smooth, differentiable evaluation.  evalW(t)
 * returns p_t(t) and its first two derivatives.
 *
 * This example constructs a DenominatorDensity for H2O and evaluates the
 * density on a uniform t-grid, printing a table suitable for plotting.
 *
 * Building
 * --------
 * From the examples/cpp/02_denominator_density/ directory:
 *
 *   cmake -B build -S .
 *   cmake --build build
 *   ./build/denominator_density
 *
 * See 03_biased_vs_unbiased/ for how this density is used in the optimisation.
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include "minimax_cpppy/denominator_density.hpp"


// =============================================================================
// H2O RHF/cc-pVDZ orbital energies (Hartree)
// Same values used in the test suite.
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


int main()
{
    // =========================================================================
    // 1. Construct DenominatorDensity
    // =========================================================================
    //
    // Parameters:
    //   occ, n_occ     — occupied orbital energies (sorted ascending)
    //   virt, n_virt   — virtual orbital energies  (sorted ascending)
    //   bandwidth      — KDE smoothing: sigma = bandwidth * ln(ratio) * delta_min
    //   n_fft          — FFT grid size (default 4096); increase for very large ratio
    //   n_t            — number of spline knots (default 512)
    //   floor_frac     — p_t floor as fraction of peak (default 1e-3)
    //
    // After construction:
    //   .deltaMin()  = 2*(eps_virt_min - eps_occ_max) = smallest pairwise Delta
    //   .deltaMax()  = 2*(eps_virt_max - eps_occ_min) = largest  pairwise Delta
    //   .ratio()     = deltaMax / deltaMin

    minimax_cpppy::DenominatorDensity d(
        H2O_OCC,  N_OCC,
        H2O_VIRT, N_VIRT,
        1.0    /* bandwidth */);

    std::printf("=============================================================\n");
    std::printf("DenominatorDensity for H2O (RHF/cc-pVDZ, bandwidth=1.0)\n");
    std::printf("=============================================================\n");
    std::printf("  delta_min = %12.6f Eh   (smallest pairwise denominator)\n",
                d.deltaMin());
    std::printf("  delta_max = %12.6f Eh   (largest  pairwise denominator)\n",
                d.deltaMax());
    std::printf("  ratio     = %12.6f       (delta_max / delta_min)\n",
                d.ratio());
    std::printf("  ln(ratio) = %12.6f       (log-space width of interval)\n",
                std::log(d.ratio()));
    std::printf("\n");


    // =========================================================================
    // 2. Evaluate density on uniform t-grid
    // =========================================================================
    //
    // t = ln(Delta / delta_min) maps Delta in [delta_min, delta_max] to [0, ln(ratio)].
    //
    // evalW(t, w, dw, d2w):
    //   w   = p_t(t)     — probability density in log-denominator space
    //   dw  = p_t'(t)    — first derivative  (used by Remez algorithm)
    //   d2w = p_t''(t)   — second derivative (used by Remez algorithm)
    //
    // The density integrates to 1 over [0, ln(ratio)].
    // High w(t) means many orbital pairs contribute denominators near exp(t)*delta_min.

    const int    n_grid = 24;
    const double t_max  = std::log(d.ratio());

    std::printf("=============================================================\n");
    std::printf("Density p_t(t) on uniform t-grid (n=%d)\n", n_grid);
    std::printf("  t=0: small denominators (near HOMO-LUMO gap)\n");
    std::printf("  t=ln(ratio): large denominators (spectral extremes)\n");
    std::printf("=============================================================\n");
    std::printf("  %8s   %12s   %14s   %14s   bar\n",
                "t", "Delta (Eh)", "p_t(t)", "p_t'(t)");
    std::printf("  %8s   %12s   %14s   %14s   %s\n",
                "--------", "------------", "--------------",
                "--------------", "----------------------------");

    // Find peak for bar scaling
    double w_max = 0.0;
    for (int i = 0; i <= 200; ++i) {
        double t = t_max * i / 200.0;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        if (w > w_max) w_max = w;
    }

    const int bar_width = 28;
    for (int i = 0; i < n_grid; ++i) {
        double t     = t_max * i / (n_grid - 1);
        double delta = d.deltaMin() * std::exp(t);
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        int bar_len = static_cast<int>(w / w_max * bar_width + 0.5);
        std::printf("  %8.4f   %12.6f   %14.6e   %14.6e   ",
                    t, delta, w, dw);
        for (int b = 0; b < bar_len; ++b) std::printf("\xe2\x96\x88");
        std::printf("\n");
    }
    std::printf("\n");


    // =========================================================================
    // 3. Locate the density peak
    // =========================================================================
    //
    // The peak marks the most common denominator — the region where quadrature
    // accuracy matters most for the correlation energy.

    double t_star = 0.0, w_star = 0.0;
    for (int i = 0; i <= 5000; ++i) {
        double t = t_max * i / 5000.0;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        if (w > w_star) { w_star = w; t_star = t; }
    }

    std::printf("=============================================================\n");
    std::printf("Density peak\n");
    std::printf("=============================================================\n");
    std::printf("  Peak at  t*     = %10.6f  (t_max = %.6f)\n",
                t_star, t_max);
    std::printf("           Delta* = %10.6f Eh = delta_min * exp(t*)\n",
                d.deltaMin() * std::exp(t_star));
    std::printf("           p_t*   = %10.6e\n", w_star);
    std::printf("\n");
    std::printf("  Normalised position: t*/t_max = %.3f\n", t_star / t_max);
    std::printf("  Biased optimisation concentrates quadrature nodes near Delta*.\n");
    std::printf("\nDone.  See 03_biased_vs_unbiased/ for the optimisation comparison.\n");

    return 0;
}
