// include/minimax_cpppy/biasing.hpp
#pragma once
#include <iosfwd>
#include "minimax_cpppy/minimax.hpp"
#include "minimax_cpppy/denominator_density.hpp"

namespace minimax_cpppy {

// Density-uncorrelated bias correction: drives the density-weighted net signed
// bias r(theta) = int_1^R eta(x)p(x)dx toward zero, warm-started from the
// unbiased minimax. Builds its own DenominatorDensity internally from
// (occ, virt, bandwidth, ...) -- its ratio() must match ymax/ymin within 1 ppm
// (skipped for near-degenerate ratios).
// errmax follows the same normalised-domain convention as laplaceMinimax
// (NOT the achieved r(theta), which is a different, smaller quantity by
// design).
MinimaxResult biasedLaplace(
    int nlap, double ymin, double ymax,
    const double* occ, int n_occ,
    const double* virt, int n_virt,
    double bandwidth,
    int n_fft = 4096, int n_t = 512,
    double floor_frac = 1e-3, double floor_frac_max = -1.0,
    double C = 0.0, int n_exc = 2,
    int verbose = 3, std::ostream& os = std::cerr);

} // namespace minimax_cpppy
