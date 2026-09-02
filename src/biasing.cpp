#include "minimax_cpppy/biasing.hpp"
#include "core/density.hpp"
#include <stdexcept>
#include <string>

namespace minimax_cpppy {

DenominatorDensity::DenominatorDensity(
    const double* occ,  int n_occ,
    const double* virt, int n_virt,
    double bandwidth,
    int    n_fft,
    int    n_t,
    double floor_frac,
    double floor_frac_max,
    double C,
    int    n_exc)
{
    if (n_occ < 1)
        throw std::invalid_argument("DenominatorDensity: n_occ must be >= 1");
    if (n_virt < 1)
        throw std::invalid_argument("DenominatorDensity: n_virt must be >= 1");
    if (!occ || !virt)
        throw std::invalid_argument("DenominatorDensity: occ and virt must not be null");
    if (bandwidth <= 0.0)
        throw std::invalid_argument(
            "DenominatorDensity: bandwidth must be > 0, got " +
            std::to_string(bandwidth));
    if (n_fft < 1 || (n_fft & (n_fft - 1)) != 0)
        throw std::invalid_argument(
            "DenominatorDensity: n_fft must be a positive power of two, got " +
            std::to_string(n_fft));

    if (n_exc < 1)
        throw std::invalid_argument(
            "DenominatorDensity: n_exc must be >= 1, got " +
            std::to_string(n_exc));
    if (C < 0.0)
        throw std::invalid_argument(
            "DenominatorDensity: C must be >= 0, got " + std::to_string(C));

    auto arr = detail::buildDensityArrays(occ, n_occ, virt, n_virt,
                                          bandwidth, n_fft, n_t, floor_frac,
                                          floor_frac_max, n_exc);
    t_knots_   = std::move(arr.logGridKnots);
    w_knots_   = std::move(arr.weightKnots);
    dw_knots_  = std::move(arr.weightDerivKnots);
    d2w_knots_ = std::move(arr.weightSecondDerivKnots);
    ratio_     = arr.ratio;
    delta_min_ = arr.denominatorMin;
    delta_max_ = arr.denominatorMax;
    t_max_     = arr.logRatioMax;
    density_offset_ = C / t_max_;
}

} // namespace minimax_cpppy
