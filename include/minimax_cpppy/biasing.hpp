#pragma once
#include <algorithm>
#include <cmath>
#include <iosfwd>
#include <stdexcept>
#include <vector>
#include "minimax_cpppy/minimax.hpp"

namespace minimax_cpppy {

class DenominatorDensity {
public:
    DenominatorDensity(
        const double* occ,  int n_occ,
        const double* virt, int n_virt,
        double bandwidth,
        int    n_fft          = 4096,
        int    n_t            = 512,
        double floor_frac     = 1e-3,
        double floor_frac_max = -1.0,
        double C              = 0.0, ///< constant density pedestal in units of the uniform density (1/t_max); must be ≥ 0
        int    n_exc          = 2); ///< excitation order: 1=singles, 2=doubles, 3=triples, …; must be ≥ 1

    void evalW(double t, double& w, double& dw, double& d2w) const noexcept;

    double ratio()    const noexcept { return ratio_; }
    double deltaMin() const noexcept { return delta_min_; }
    double deltaMax() const noexcept { return delta_max_; }

private:
    std::vector<double> t_knots_;
    std::vector<double> w_knots_;
    std::vector<double> dw_knots_;
    std::vector<double> d2w_knots_;
    double ratio_     = 0.0;
    double delta_min_ = 0.0;
    double delta_max_ = 0.0;
    double t_max_     = 0.0;
    double density_offset_ = 0.0; ///< absolute pedestal added to w in evalW (= C / t_max_)
};

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

inline void DenominatorDensity::evalW(double t, double& w, double& dw, double& d2w)
    const noexcept
{
    double tc = t < 0.0 ? 0.0 : (t > t_max_ ? t_max_ : t);

    const int n = static_cast<int>(t_knots_.size()) - 1;
    int j = static_cast<int>(
        std::lower_bound(t_knots_.begin(), t_knots_.end(), tc) - t_knots_.begin()) - 1;
    j = j < 0 ? 0 : (j > n-1 ? n-1 : j);

    double h  = t_knots_[j+1] - t_knots_[j];
    double s  = (tc - t_knots_[j]) / h;
    double s2 = s*s, s3 = s2*s, s4 = s3*s, s5 = s4*s;

    double H0 = 1.0 - 10*s3 + 15*s4 -  6*s5;
    double H1 =         s - 6*s3 + 8*s4 - 3*s5;
    double H2 = 0.5*s2 - 1.5*s3 + 1.5*s4 - 0.5*s5;
    double H3 =         10*s3 - 15*s4 + 6*s5;
    double H4 =         -4*s3 +  7*s4 - 3*s5;
    double H5 = 0.5*s3 - s4 + 0.5*s5;

    double hdf0  = h * dw_knots_[j],    hdf1  = h * dw_knots_[j+1];
    double h2d0  = h*h*d2w_knots_[j],   h2d1  = h*h*d2w_knots_[j+1];

    w = H0*w_knots_[j] + H1*hdf0 + H2*h2d0 + H3*w_knots_[j+1] + H4*hdf1 + H5*h2d1;
    if (w < 0.0) { w = density_offset_; dw = 0.0; d2w = 0.0; return; }

    double dH0 = -30*s2 + 60*s3 - 30*s4;
    double dH1 =   1.0  - 18*s2 + 32*s3 - 15*s4;
    double dH2 =   s    - 4.5*s2 + 6*s3 - 2.5*s4;
    double dH3 =  30*s2 - 60*s3 + 30*s4;
    double dH4 = -12*s2 + 28*s3 - 15*s4;
    double dH5 =  1.5*s2 - 4*s3 + 2.5*s4;

    double dyds = dH0*w_knots_[j] + dH1*hdf0 + dH2*h2d0
                + dH3*w_knots_[j+1] + dH4*hdf1 + dH5*h2d1;
    dw = dyds / h;

    double d2H0 = -60*s  + 180*s2 - 120*s3;
    double d2H1 = -36*s  +  96*s2 -  60*s3;
    double d2H2 =   1.0  -   9*s  +  18*s2 - 10*s3;
    double d2H3 =  60*s  - 180*s2 + 120*s3;
    double d2H4 = -24*s  +  84*s2 -  60*s3;
    double d2H5 =   3*s  -  12*s2 +  10*s3;

    double d2yds2 = d2H0*w_knots_[j] + d2H1*hdf0 + d2H2*h2d0
                  + d2H3*w_knots_[j+1] + d2H4*hdf1 + d2H5*h2d1;
    d2w = d2yds2 / (h*h);

    w += density_offset_;
}

} // namespace minimax_cpppy
