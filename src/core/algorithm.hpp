/**
 * @file algorithm.hpp
 * @brief MINIMAX Laplace quadrature: public C++ dispatcher.
 *
 * Thin dispatcher that exposes the two `laplaceMinimax` overloads and
 * transitively pulls in the full algorithm via:
 *   algorithm.hpp → remez.hpp → paraopt.hpp → maehly.hpp → dd128.hpp
 *
 * Approximates 1/x ≈ Σ_k w_k exp(−a_k x) for x ∈ [ymin, ymax] with
 * minimax (equioscillation) optimal exponents and weights.
 * Internally uses 128-bit double-double (DD) arithmetic for the Newton-Maehly
 * extremum search and the Newton-Raphson equioscillation solver.
 */
#pragma once
#ifndef MINIMAX_CPPPY_MINIMAX_IMPL_HPP
#define MINIMAX_CPPPY_MINIMAX_IMPL_HPP

#include <cmath>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "remez.hpp"

namespace minimax_cpppy {
namespace detail {

/**
 * @brief Compute MINIMAX-optimal Laplace exponents and weights for 1/x on [ymin, ymax].
 *
 * Initialises from the dense interpolated table for any nlap in [1, MAX_NLAP].
 * Verifies table quality by sampling; falls back to logspace LS init for stale
 * or copied table entries.  Passes interpolated extrema as warm-start when valid.
 *
 * @param nlap            Number of Laplace quadrature points; must satisfy 1 <= nlap <= 30.
 * @param ymin            Lower bound of the approximation interval (must be > 0).
 * @param ymax            Upper bound of the approximation interval (must be > ymin).
 * @param maxIter         Maximum Remez outer iterations (default 200).
 * @param toleranceMaehly Newton-Maehly convergence tolerance (default 1e-10).
 * @param toleranceNR     Newton-Raphson convergence tolerance (default 1e-15).
 * @param stepMax         Trust-region step cap for both inner solvers (default 0.3).
 * @param delta           Relative shift for initial extremum guess in maehlySolver (default 1e-6).
 * @param armijoConstant  Armijo constant for the inner line search (default 1e-4).
 * @return minimax_cpppy::MinimaxResult with expon, weight, and errmax.
 * @note   Four dispatch paths based on sampled approximation error:
 *         (1) sampledErr < 1e-12 — table holds machine-precision solution, return
 *             directly (Remez/Maehly would fail on a flat error function);
 *         (2) sampledErr ≤ 0.5 and extrema in range with sufficient span — valid
 *             table init with extrema warm-start;
 *         (3) sampledErr ≤ 0.5 but extrema out-of-range or span < sqrt(ratio) —
 *             logspace LS attempted; if LS error < 1% return directly (over-
 *             parameterised), else cold Maehly with table init;
 *         (4) sampledErr > 0.5 — table is stale copy, same LS dispatch as (3)
 *             but also falls back to logspace(10.0) seed for remezLoop.
 * @throws std::invalid_argument if nlap or energy range are invalid.
 * @throws std::runtime_error if no tabulated initial data is available.
 */
inline minimax_cpppy::MinimaxResult laplaceMinimax(int nlap, double ymin, double ymax,
                                     int    maxIter         = 200,
                                     double toleranceMaehly = 1e-10,
                                     double toleranceNR     = 1e-15,
                                     double stepMax         = 0.3,
                                     double delta           = 1e-6,
                                     double armijoConstant  = 1e-4,
                                     int    verbose         = 0,
                                     std::ostream* os       = nullptr)
{
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (nlap < 1 || nlap > data::MAX_NLAP) {
        throw std::invalid_argument(
            "laplaceMinimax: nlap out of [1," + std::to_string(data::MAX_NLAP) + "]");
    }
    if (ymin <= 0.0 || ymax <= ymin) {
        throw std::invalid_argument("laplaceMinimax: need 0 < ymin < ymax");
    }
#endif

    const double ratio = ymax / ymin;

    // Rescale a normalised [1, ratio] solution back to the physical [ymin, ymax] domain.
    auto scaleResult = [&](const std::vector<DD>& exp, const std::vector<DD>& wt, DD err) {
        minimax_cpppy::MinimaxResult result;
        result.expon.resize(nlap);
        result.weight.resize(nlap);
        for (int k = 0; k < nlap; ++k) {
            result.expon[k]  = (exp[k] / ymin).hi;
            result.weight[k] = (wt[k]  / ymin).hi;
        }
        result.errmax = err.hi;
        return result;
    };

    std::vector<DD> exponents(nlap), weights(nlap);
    DD errorAmplitude;
    std::vector<double> initExtrema;

    // Over-parameterised case (small ratio, large nlap): if the log-spaced LS
    // init already hits <1% error, return it directly. Returns a filled
    // MinimaxResult on success, std::nullopt to keep the cold-Maehly path.
    auto tryOverparamReturn = [&]() -> std::optional<minimax_cpppy::MinimaxResult> {
        std::vector<DD> fbExp, fbWt;
        DD fbErr;
        if (logspaceInitFallback(fbExp, fbWt, fbErr, nlap, ratio, 0.01)) {
            return scaleResult(fbExp, fbWt, fbErr);
        }
        return std::nullopt;
    };

    {
        data::ExtTableEntry interp = data::interpolatedLookup(nlap, ratio);
        for (int k = 0; k < nlap; ++k) {
            exponents[k] = DD(interp.expon[k]);
            weights[k]   = DD(interp.weight[k]);
        }
        errorAmplitude = DD(std::abs(interp.errmax));

        // Verify table quality by sampling: a copied-neighbor entry produces
        // large actual error for the target ratio even if errmax looks fine.
        DD sampledErr = computeInitialError(exponents.data(), weights.data(),
                                            nlap, ratio, 2 * nlap + 1);

        // When sampledErr is at machine precision the table holds an essentially
        // optimal solution; running Remez would fail because Maehly cannot find
        // extrema of an already-flat error function.
        if (sampledErr.hi < 1e-12) {
            return scaleResult(exponents, weights, sampledErr);
        }

        const int n_ext = 2 * nlap - 1;
        initExtrema.resize(n_ext);
        for (int k = 0; k < n_ext; ++k) { initExtrema[k] = interp.extrema[k]; }

        if (sampledErr <= DD(0.5)) {
            // Table init is valid for this ratio; attempt extrema warm-start.
            bool extremaInRange = true;
            for (int k = 0; k < n_ext; ++k) {
                if (initExtrema[k] <= 1.0 || initExtrema[k] >= ratio) {
                    extremaInRange = false;
                    break;
                }
            }
            if (!extremaInRange) {
                // Lagrange overshoot at table boundary or stale extrema from a
                // neighbor ratio.  Try logspace direct return for over-parameterised
                // cases (small ratio, large nlap); otherwise cold Maehly with table init.
                initExtrema.clear();
                if (auto r = tryOverparamReturn()) return *r;
                // Not over-parameterised: keep table init, proceed cold.
            } else {
                // Span check: copy-from-neighbor entries have all extrema clustered
                // near x=1, so maxExt << sqrt(ratio) even though bounds pass.
                double maxExt = *std::max_element(initExtrema.begin(), initExtrema.end());
                if (maxExt < std::sqrt(ratio)) {
                    initExtrema.clear();
                    if (auto r = tryOverparamReturn()) return *r;
                    // Not over-parameterised: keep table init, drop extrema warm-start.
                }
                // else: maxExt >= sqrt(ratio) — valid entry, keep warm-start.
            }
        } else {
            // sampledErr > 0.5 — table entry is a stale copy; use logspace init.
            initExtrema.clear();
            if (auto r = tryOverparamReturn()) return *r;
            std::vector<DD> fbExp, fbWt;
            DD fbErr;
            logspaceInitFallback(fbExp, fbWt, fbErr, nlap, ratio, 10.0);
            exponents      = std::move(fbExp);
            weights        = std::move(fbWt);
            errorAmplitude = fbErr;
        }
    }

    return remezLoop(
        nlap, ymin, ratio,
        std::move(exponents), std::move(weights), errorAmplitude,
        maxIter, toleranceMaehly, toleranceNR, stepMax, delta, armijoConstant,
        initExtrema.empty() ? nullptr : &initExtrema,
        nullptr, verbose, os);
}

/**
 * @brief Compute MINIMAX-optimal Laplace exponents and weights for 1/x on [ymin, ymax],
 *        starting from a caller-supplied initial guess in the normalised [1, ratio] domain.
 *
 * The PUBLIC API is responsible for normalising the user's exponents/weights from
 * [ymin, ymax] to [1, ratio] before calling this overload.  The initial error amplitude
 * is estimated by sampling the approximation error at 2*nlap+1 log-spaced points.
 *
 * @param nlap              Number of Laplace quadrature points; must satisfy 1 <= nlap <= MAX_NLAP.
 * @param ymin              Lower bound of the approximation interval (must be > 0).
 * @param ymax              Upper bound of the approximation interval (must be > ymin).
 * @param initExpNorm       Initial exponents in the normalised [1, ratio] domain, length nlap.
 * @param initWeightNorm    Initial weights   in the normalised [1, ratio] domain, length nlap.
 * @param maxIter           Maximum Remez outer iterations (default 200).
 * @param toleranceMaehly   Newton-Maehly convergence tolerance (default 1e-10).
 * @param toleranceNR       Newton-Raphson convergence tolerance (default 1e-15).
 * @param stepMax           Trust-region step cap for both inner solvers (default 0.3).
 * @param delta             Relative shift for initial extremum guess in maehlySolver (default 1e-6).
 * @param armijoConstant    Armijo constant for the inner line search (default 1e-4).
 * @return minimax_cpppy::MinimaxResult with expon, weight, and errmax.
 * @throws std::invalid_argument if nlap, energy range, or pointers are invalid (Debug only).
 */
inline minimax_cpppy::MinimaxResult laplaceMinimax(
    int nlap, double ymin, double ymax,
    const double* initExpNorm,    // exponents in [1, ratio] domain, length nlap
    const double* initWeightNorm, // weights   in [1, ratio] domain, length nlap
    int    maxIter         = 200,
    double toleranceMaehly = 1e-10,
    double toleranceNR     = 1e-15,
    double stepMax         = 0.3,
    double delta           = 1e-6,
    double armijoConstant  = 1e-4,
    int    verbose         = 0,
    std::ostream* os       = nullptr)
{
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (nlap < 1 || nlap > data::MAX_NLAP) {
        throw std::invalid_argument(
            "laplaceMinimax: nlap out of [1," + std::to_string(data::MAX_NLAP) + "]");
    }
    if (ymin <= 0.0 || ymax <= ymin) {
        throw std::invalid_argument("laplaceMinimax: need 0 < ymin < ymax");
    }
    if (!initExpNorm || !initWeightNorm) {
        throw std::invalid_argument("laplaceMinimax: null initial guess pointer");
    }
#endif

    const double ratio = ymax / ymin;

    std::vector<DD> exponents(nlap), weights(nlap);
    for (int k = 0; k < nlap; ++k) {
        exponents[k] = DD(initExpNorm[k]);
        weights[k]   = DD(initWeightNorm[k]);
    }

    DD errorAmplitude = computeInitialError(exponents.data(), weights.data(),
                                             nlap, ratio, 8 * nlap + 1);

    return remezLoop(nlap, ymin, ratio, std::move(exponents), std::move(weights), errorAmplitude,
                     maxIter, toleranceMaehly, toleranceNR, stepMax, delta, armijoConstant,
                     nullptr, nullptr, verbose, os);
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_MINIMAX_IMPL_HPP
