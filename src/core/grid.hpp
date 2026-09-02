#ifndef MINIMAX_CPPPY_GRID_HPP
#define MINIMAX_CPPPY_GRID_HPP
#include <algorithm>
#include <cmath>
#include <vector>

namespace minimax_cpppy {
namespace detail {

// Cusp (kink of ln(value) vs ln R) in log10 R. From scratch/quad_scan/FINDINGS.md.
inline double cuspLog10(int nlap) {
    const double n = static_cast<double>(nlap);
    return (4.4877 * std::sqrt(n) - 0.3845 * std::log(n) - 2.3654) / 2.302585092994046;
}

struct GridParams {
    int n_dense;            // dense nodes on the no-polish / over-resolved side
    int n_sparse;           // sparse uniform nodes on the polished tail
    double half_width_dec;  // cluster half-width (decades) around the cusp
    double margin_dec;      // dense side extends to cusp + margin_dec
};

// Two-tier grid: dense cusp-clustered nodes on [lo, b], sparse uniform on [b, hi],
// where b = min(hi, cusp + margin). Endpoints lo and hi are always present.
inline std::vector<double> buildLog10Grid(int nlap, const GridParams& p,
                                          double lo = 0.1, double hi = 12.0) {
    const double cusp = std::max(lo, std::min(hi, cuspLog10(nlap)));
    const double b = std::min(hi, cusp + p.margin_dec);
    std::vector<double> g;

    // Dense side [lo, b] covers the over-resolved span, where the returned points
    // come from direct interpolation with NO Remez polish — so it needs EVEN
    // coverage across the whole span, not a spike at the cusp. (A cusp-only
    // cluster starves the low-R end, which for high nlap is entirely
    // over-resolved and then interpolates to garbage.) Use a uniform dense grid.
    const int nd = std::max(2, p.n_dense);
    for (int i = 0; i < nd; ++i) g.push_back(lo + (b - lo) * i / (nd - 1));

    // Plus a modest refinement clustered at the cusp/kink, where the points have
    // a derivative kink (the quad_fit cusp knowledge): tighten interpolation there.
    const int nc = std::max(0, nd / 4);
    for (int i = 0; i < nc; ++i) {
        const double u = (nc > 1) ? (-1.0 + 2.0 * i / (nc - 1)) : 0.0;
        const double mag = std::pow(std::abs(u), 1.7) * p.half_width_dec;
        double x = cusp + (u < 0 ? -mag : mag);
        if (x > lo && x < b) g.push_back(x);
    }
    g.push_back(lo);
    g.push_back(b);

    // Sparse tail (b, hi].
    const int ns = std::max(0, p.n_sparse);
    for (int i = 1; i <= ns; ++i) g.push_back(b + (hi - b) * i / ns);
    g.push_back(hi);

    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end(),
            [](double a, double c){ return std::abs(a - c) < 1e-9; }), g.end());
    return g;
}

} // namespace detail
} // namespace minimax_cpppy
#endif
