/**
 * @file maehly.hpp
 * @brief Newton-Maehly extremum search for the MINIMAX Remez algorithm.
 *
 * Contains the log-space error derivative evaluators and the Newton-Maehly
 * root-finding solver used to locate equioscillation extrema of the
 * approximation error e(x) = 1/x − Σ_k w_k exp(−a_k x).
 */
#pragma once
#ifndef MINIMAX_CPPPY_MAEHLY_HPP
#define MINIMAX_CPPPY_MAEHLY_HPP

#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <vector>
#include "dd128.hpp"
#include "minimax_cpppy/minimax.hpp"

namespace minimax_cpppy {
namespace detail {

// Shared implementation: takes precomputed x = exp(t).
// Also computes unweightedError = 1/x - sum_k w_k * exp(-a_k * x), reusing exp(-u_k) from
// the hprime/h2prime loop to avoid a second pass over the same exponentials.
static void evalLogErrorDerivativesFromX(
    DD& hprime, DD& h2prime, DD& unweightedError,
    const DD& x,
    const DD* exponents, const DD* weights, int nlap) noexcept
{
    DD invx = DD(1.0) / x;
    hprime          = -invx;
    h2prime         =  invx;
    unweightedError =  invx;
    for (int k = 0; k < nlap; ++k) {
        DD u   = exponents[k] * x;
        DD eu  = DD::ddExp(-u);
        DD wue = weights[k] * u * eu;
        hprime          += wue;
        h2prime         += wue * (DD(1.0) - u);
        unweightedError -= weights[k] * eu;
    }
}

/**
 * @brief Evaluate h'(t) and h''(t) for h(t) = e(exp(t)), where e(x) = 1/x − Σ w_k exp(−a_k x).
 *
 * Using the substitution t = log(x), x = exp(t), u_k = a_k * x:
 *   h'(t)  = −1/x + Σ w_k u_k exp(−u_k)
 *   h''(t) =  1/x + Σ w_k u_k (1 − u_k) exp(−u_k)
 *
 * Underflow-safe: when u_k > 512, DD::ddExp(−u_k) returns DD(0,0); term contribution is 0.
 *
 * Used by the log-space Newton-Maehly extremum search.
 *
 * @param hprime    Output: h'(t) at the given t.
 * @param h2prime   Output: h''(t) at the given t.
 * @param t         Log-space evaluation point (t = log(x)).
 * @param exponents Current exponents a_k, length nlap.
 * @param weights   Current weights w_k, length nlap.
 * @param nlap      Number of Laplace points.
 */
static void evalLogErrorDerivatives(DD& hprime, DD& h2prime,
                                    const DD& t,
                                    const DD* exponents, const DD* weights,
                                    int nlap)
#ifndef MINIMAX_CPPPY_DEBUG_MODE__
noexcept
#endif
{
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    ++minimax_cpppy::debug::functional1_call_count;
    if (!exponents || !weights) {
        throw std::invalid_argument("evalLogErrorDerivatives: null pointer argument");
    }
    if (nlap <= 0) {
        throw std::invalid_argument("evalLogErrorDerivatives: non-positive nlap");
    }
    if (!std::isfinite(t.hi)) {
        throw std::invalid_argument("evalLogErrorDerivatives: t is not finite");
    }
#endif
    DD x = DD::ddExp(t);
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (x.hi <= 0.0 || !std::isfinite(x.hi)) {
        throw std::runtime_error(
            "evalLogErrorDerivatives: exp(t) is non-positive or non-finite");
    }
#endif
    DD dummy;
    evalLogErrorDerivativesFromX(hprime, h2prime, dummy, x, exponents, weights, nlap);
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (!std::isfinite(hprime.hi) || !std::isfinite(h2prime.hi)) {
        throw std::runtime_error(
            "evalLogErrorDerivatives: NaN/Inf in output (hprime or h2prime)");
    }
#endif
}

/**
 * @brief Find numExtrema interior extrema of the error derivative e'(x) via Newton + Maehly deflation.
 *
 * Uses Newton's method with Maehly deflation to find successive roots of e'(x),
 * which are the equioscillation extrema required by the Remez algorithm.
 *
 * @param extremaOut    Output array of numExtrema extremum locations; sorted ascending on return.
 * @param numExtrema    Number of interior extrema to find.
 * @param rangeLo       Lower bound of search interval (normalised, typically 1.0).
 * @param rangeHi       Upper bound of search interval (= ymax/ymin).
 * @param maxIter       Maximum Newton iterations per root.
 * @param tolerance     Convergence tolerance |dx| < tolerance.
 * @param exponents     Current exponents a_k, length nlap.
 * @param weights       Current weights w_k, length nlap.
 * @param nlap          Number of Laplace points.
 * @param stepMax       Maximum Newton step size (trust-region cap).
 * @param delta         Additive log-space shift for initial guess of next root.
 */
static void maehlySolver(DD* extremaOut, int numExtrema,
                          DD rangeLo, DD rangeHi,
                          int maxIter, double tolerance,
                          const DD* exponents, const DD* weights, int nlap,
                          double stepMax, double delta,
                          const double* initialGuess = nullptr,
                          int verbose = 0, std::ostream* os = nullptr)
{
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (!extremaOut || !exponents || !weights) {
        throw std::invalid_argument("maehlySolver: null pointer argument");
    }
    if (nlap <= 0 || numExtrema <= 0) {
        throw std::invalid_argument("maehlySolver: non-positive nlap or numExtrema");
    }
    if (rangeHi.hi <= rangeLo.hi) {
        throw std::invalid_argument("maehlySolver: rangeHi <= rangeLo");
    }
#endif
    const DD toleranceDD(tolerance);

    const DD logRangeLo(0.0);
    const DD logRangeHi = DD::ddLog(rangeHi);
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (logRangeHi.hi <= 0.0 || !std::isfinite(logRangeHi.hi)) {
        throw std::runtime_error(
            "maehlySolver: log(rangeHi) is non-positive or non-finite");
    }
#endif

    std::vector<DD> extremaOut_t(numExtrema);
    DD startPoint_t = logRangeLo;

    for (int rootIdx = 0; rootIdx < numExtrema; ++rootIdx) {
        DD stepMaxPerRoot_t;
        if (rootIdx == 0) {
            stepMaxPerRoot_t = DD(stepMax);
        } else if (rootIdx == 1) {
            stepMaxPerRoot_t = DD(stepMax) * extremaOut_t[0];
        } else {
            stepMaxPerRoot_t = DD(stepMax) * (extremaOut_t[rootIdx-1] - extremaOut_t[rootIdx-2]);
        }

        DD currentPoint_t;
        if (initialGuess != nullptr) {
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
            if (initialGuess[rootIdx] <= 0.0 || !std::isfinite(initialGuess[rootIdx])) {
                throw std::invalid_argument(
                    "maehlySolver: initialGuess entry is non-positive or non-finite");
            }
#endif
            DD rawT = DD::ddLog(DD(initialGuess[rootIdx]));
            if (rawT >= logRangeHi || rawT <= logRangeLo) {
                DD last_t = (rootIdx > 0) ? extremaOut_t[rootIdx - 1] : logRangeLo;
                int remaining = numExtrema - rootIdx;
                currentPoint_t = last_t + (logRangeHi - last_t) / DD(remaining + 1);
                if (verbose >= 2 && os) {
                    *os << "[Maehly WARN root=" << rootIdx << "] hint t=" << rawT.hi
                        << " outside domain; using adaptive t=" << currentPoint_t.hi << '\n';
                }
            } else {
                currentPoint_t = rawT;
            }
        } else {
            currentPoint_t = startPoint_t;
        }
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
        if (!std::isfinite(currentPoint_t.hi)) {
            throw std::runtime_error(
                "maehlySolver: initial currentPoint_t is non-finite");
        }
#endif

        DD lastHprime(0.0), lastStep(0.0);
        bool converged = false;
        for (int it = 0; it < maxIter; ++it) {
            DD hprime, h2prime;
            evalLogErrorDerivatives(hprime, h2prime, currentPoint_t,
                                    exponents, weights, nlap);

            DD deflationSum(0.0);
            for (int j = 0; j < rootIdx; ++j) {
                deflationSum += DD(1.0) / (currentPoint_t - extremaOut_t[j]);
            }
            h2prime -= deflationSum * hprime;

#ifdef MINIMAX_CPPPY_DEBUG_MODE__
            if (!std::isfinite(h2prime.hi) || h2prime.hi == 0.0) {
                throw std::runtime_error(
                    "maehlySolver: h2prime is zero or non-finite before Newton step");
            }
#endif
            DD step = hprime / h2prime;
            if (DD::ddAbs(step) > stepMaxPerRoot_t) {
                step = stepMaxPerRoot_t * (step.hi >= 0.0 ? DD(1.0) : DD(-1.0));
            }
            if (verbose >= 4 && os) {
                *os << "[Maehly root=" << rootIdx << " iter=" << it
                    << "] t=" << currentPoint_t.hi
                    << " hprime=" << hprime.hi
                    << " step=" << step.hi << '\n';
            }
            lastHprime = hprime;
            lastStep   = step;
            currentPoint_t -= step;
            if (currentPoint_t >= logRangeHi) currentPoint_t = logRangeHi - toleranceDD;
            if (currentPoint_t <= logRangeLo) currentPoint_t = logRangeLo + toleranceDD;
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
            if (!std::isfinite(currentPoint_t.hi)) {
                throw std::runtime_error(
                    "maehlySolver: currentPoint_t became non-finite after Newton step");
            }
#endif
            if (DD::ddAbs(step) < toleranceDD) { converged = true; break; }
        }
        if (!converged) {
            if (verbose >= 1 && os) {
                *os << "[Maehly ERROR root=" << rootIdx << "] did not converge"
                    << " t=" << currentPoint_t.hi
                    << " hprime=" << lastHprime.hi
                    << " lastStep=" << lastStep.hi << '\n';
            }
            throw std::runtime_error("minimax: Newton-Maehly did not converge");
        }

        for (int it = 0; it < maxIter; ++it) {
            DD hprime, h2prime;
            evalLogErrorDerivatives(hprime, h2prime, currentPoint_t,
                                    exponents, weights, nlap);
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
            if (!std::isfinite(h2prime.hi) || h2prime.hi == 0.0) {
                throw std::runtime_error(
                    "maehlySolver: h2prime zero or non-finite in polish step");
            }
#endif
            DD step = hprime / h2prime;
            if (DD::ddAbs(step) > stepMaxPerRoot_t) {
                step = stepMaxPerRoot_t * (step.hi >= 0.0 ? DD(1.0) : DD(-1.0));
            }
            if (verbose >= 4 && os) {
                *os << "[Maehly polish root=" << rootIdx << " iter=" << it
                    << "] t=" << currentPoint_t.hi
                    << " hprime=" << hprime.hi
                    << " step=" << step.hi << '\n';
            }
            currentPoint_t -= step;
            if (currentPoint_t >= logRangeHi) currentPoint_t = logRangeHi - toleranceDD;
            if (currentPoint_t <= logRangeLo) currentPoint_t = logRangeLo + toleranceDD;
            if (DD::ddAbs(step) < toleranceDD) { break; }
        }

        extremaOut_t[rootIdx] = currentPoint_t;

        {
            const int remaining_roots = numExtrema - rootIdx - 1;
            DD proposed = currentPoint_t + DD(delta);
            if (remaining_roots > 0 && proposed >= logRangeHi) {
                startPoint_t = currentPoint_t
                               + (logRangeHi - currentPoint_t) / DD(remaining_roots + 1);
                if (verbose >= 2 && os) {
                    *os << "[Maehly INFO root=" << rootIdx << "] adaptive startPoint="
                        << startPoint_t.hi << " (" << remaining_roots
                        << " roots remain)\n";
                }
            } else {
                startPoint_t = proposed;
                if (startPoint_t < logRangeLo) {
                    if (verbose >= 2 && os) {
                        *os << "[Maehly WARN root=" << rootIdx << "] startPoint clamped to logRangeLo="
                            << logRangeLo.hi << " (was " << startPoint_t.hi << ")\n";
                    }
                    startPoint_t = logRangeLo;
                }
            }
        }
    }

    for (int i = 0; i < numExtrema; ++i) {
        extremaOut[i] = DD::ddExp(extremaOut_t[i]);
    }

    std::sort(extremaOut, extremaOut + numExtrema,
              [](const DD& a, const DD& b){ return a < b; });

    if (!(extremaOut[0] > rangeLo)) {
        if (verbose >= 1 && os) {
            *os << "[Maehly ERROR] lowest extremum=" << extremaOut[0].hi
                << " <= rangeLo=" << rangeLo.hi << '\n';
        }
        throw std::runtime_error("minimax: lowest extremum at or below range lo");
    }
    if (!(extremaOut[numExtrema-1] < rangeHi)) {
        if (verbose >= 1 && os) {
            *os << "[Maehly ERROR] highest extremum[" << (numExtrema - 1) << "]="
                << extremaOut[numExtrema-1].hi
                << " >= rangeHi=" << rangeHi.hi << '\n';
        }
        throw std::runtime_error("minimax: highest extremum at or above range hi");
    }
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_MAEHLY_HPP
