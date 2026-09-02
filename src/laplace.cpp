#include "minimax_cpppy/minimax.hpp"
#include "minimax_cpppy/biasing.hpp"
#include "core/algorithm.hpp"
#include "core/algorithm_biased.hpp"
#include "core/log_pretty.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax_cpppy {

static void validateLaplaceArgs(int nlap, double ymin, double ymax) {
    if (nlap < 1 || nlap > detail::data::MAX_NLAP)
        throw std::invalid_argument(
            "laplaceMinimax: nlap must be in [1, " +
            std::to_string(detail::data::MAX_NLAP) + "], got " +
            std::to_string(nlap));
    if (ymin <= 0.0)
        throw std::invalid_argument(
            "laplaceMinimax: ymin must be > 0, got " + std::to_string(ymin));
    if (ymax <= ymin)
        throw std::invalid_argument(
            "laplaceMinimax: ymax must be > ymin; got ymin=" +
            std::to_string(ymin) + " ymax=" + std::to_string(ymax));
}

MinimaxResult laplaceMinimax(int nlap, double ymin, double ymax,
                              int verbose, std::ostream& os) {
    validateLaplaceArgs(nlap, ymin, ymax);
    if (verbose >= 1) detail::printRunBanner(os, "laplaceMinimax", nlap, ymin, ymax);
    MinimaxResult result = detail::laplaceMinimax(nlap, ymin, ymax,
                                  200, 1e-10, 1e-15, 0.3, 1e-6, 1e-4, verbose, &os);
    if (verbose >= 3) detail::printCitationBlock(os);
    return result;
}

MinimaxResult laplaceMinimax(int nlap, double ymin, double ymax,
                              const std::vector<double>& initExponents,
                              const std::vector<double>& initWeights,
                              int verbose, std::ostream& os)
{
    validateLaplaceArgs(nlap, ymin, ymax);
    if (static_cast<int>(initExponents.size()) != nlap ||
        static_cast<int>(initWeights.size()) != nlap) {
        throw std::invalid_argument(
            "laplaceMinimax: initExponents and initWeights must have exactly nlap=" +
            std::to_string(nlap) + " elements");
    }
    if (verbose >= 1) detail::printRunBanner(os, "laplaceMinimax", nlap, ymin, ymax);
    // Normalise from physical [ymin, ymax] to [1, ratio]: a_norm = a_phys * ymin
    // (x_phys = x_norm / ymin, so a_phys * x_phys = a_norm * x_norm with a_norm = a_phys * ymin)
    std::vector<double> normExponents(nlap), normWeights(nlap);
    for (int k = 0; k < nlap; ++k) {
        normExponents[k] = initExponents[k] * ymin;
        normWeights[k]   = initWeights[k]   * ymin;
    }
    MinimaxResult result = detail::laplaceMinimax(nlap, ymin, ymax,
                                  normExponents.data(), normWeights.data(),
                                  200, 1e-10, 1e-15, 0.3, 1e-6, 1e-4, verbose, &os);
    if (verbose >= 3) detail::printCitationBlock(os);
    return result;
}

MinimaxResult biasedLaplace(
    int nlap, double ymin, double ymax,
    const double* occ, int n_occ,
    const double* virt, int n_virt,
    double bandwidth, int n_fft, int n_t,
    double floor_frac, double floor_frac_max, double C, int n_exc,
    int verbose, std::ostream& os)
{
    validateLaplaceArgs(nlap, ymin, ymax);
    if (verbose >= 1) detail::printRunBanner(os, "biasedLaplace", nlap, ymin, ymax);
    MinimaxResult result = detail::biasedLaplace(
        nlap, ymin, ymax, occ, n_occ, virt, n_virt,
        bandwidth, n_fft, n_t, floor_frac, floor_frac_max, C, n_exc,
        200, 1e-10, 1e-15, 0.3, 1e-6, 1e-4,
        verbose, &os);
    if (verbose >= 3) detail::printCitationBlock(os);
    return result;
}

} // namespace minimax_cpppy
