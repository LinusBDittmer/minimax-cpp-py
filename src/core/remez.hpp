/**
 * @file remez.hpp
 * @brief Remez outer loop and initialisation helpers for MINIMAX Laplace quadrature.
 *
 * Contains the initial error sampler, the log-spaced LS fallback initialiser,
 * and the main Remez iteration loop that alternates between the Newton-Maehly
 * extremum search and the Newton-Raphson equioscillation solver.
 */
#pragma once
#ifndef MINIMAX_CPPPY_REMEZ_HPP
#define MINIMAX_CPPPY_REMEZ_HPP

#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "paraopt.hpp"
#include "log_pretty.hpp"
#include "data_ext.hpp"

namespace minimax_cpppy {
namespace detail {

/**
 * @brief Sample the maximum approximation error |1/x - Σ w_k exp(-a_k x)| at
 *        nsamp log-spaced points on [1, ratio] using DD arithmetic.
 *
 * Used to seed the Remez loop when the caller supplies their own initial guess
 * instead of tabulated data.
 *
 * @param exponents Exponents a_k in the normalised [1, ratio] domain, length nlap.
 * @param weights   Weights w_k in the normalised [1, ratio] domain, length nlap.
 * @param nlap      Number of Laplace points.
 * @param ratio     Upper bound of normalised interval (= ymax / ymin).
 * @param nsamp     Number of log-spaced sample points.
 * @return Maximum sampled |e(x)|.
 */
static DD computeInitialError(const DD* exponents, const DD* weights,
                               int nlap, double ratio, int nsamp)
{
    DD logRatio(std::log(ratio));
    DD maxError(0.0);
    for (int i = 0; i < nsamp; ++i) {
        double t = static_cast<double>(i) / (nsamp - 1);
        DD x = DD::ddExp(DD(t) * logRatio);   // x = ratio^t, covers [1, ratio] log-spaced
        DD approxVal(0.0);
        for (int k = 0; k < nlap; ++k) {
            approxVal += weights[k] * DD::ddExp(-(exponents[k] * x));
        }
        DD pointError = DD::ddAbs(DD(1.0) / x - approxVal);
        if (maxError < pointError) maxError = pointError;
    }
    return maxError;
}

/**
 * @brief Fallback init for small-ratio / large-nlap: log-spaced exponents + LS weights.
 *
 * Places nlap exponents as interior points of a log-uniform grid of size nlap+2
 * on [1, ratio]:  a_k = ratio^((k+1)/(nlap+1)), k = 0..nlap-1.
 * All points are strictly inside (1, ratio).
 *
 * Weights are found in two stages:
 *   (a) A least-squares solve with strong Tikhonov regularisation
 *       (lambda = 1e-4 * max diag of A^T A) to get a safe starting point.
 *   (b) A linear Remez loop (up to 50 iters) that fixes exponents and
 *       optimises only the nlap weights + error amplitude ε.  Each iteration
 *       solves the (nlap+1)×(nlap+1) Chebyshev equioscillation linear system
 *       at the current nlap+1 alternation points, then updates those points
 *       via Newton–Maehly (nlap-1 interior extrema).
 *
 * Returns true and populates exponents, weights, errorAmplitude if the
 * resulting max |e(x)| over 8*nlap sample points is <= maxAcceptableError.
 * Returns false otherwise.
 *
 * @param exponents          Out: log-spaced exponents in [1, ratio] domain.
 * @param weights            Out: LS weights in [1, ratio] domain.
 * @param errorAmplitude     Out: max sampled |1/x - sum_k w_k exp(-a_k x)|.
 * @param nlap               Number of quadrature points (>= 1).
 * @param ratio              ymax/ymin (must be > 1).
 * @param maxAcceptableError Threshold for acceptable max error.
 */
static bool logspaceInitFallback(
    std::vector<DD>& exponents,
    std::vector<DD>& weights,
    DD& errorAmplitude,
    int nlap, double ratio,
    double maxAcceptableError = 0.01)
{
    const double logRatio = std::log(ratio);
    const int    m        = 8 * nlap;

    // ---- 1. Log-spaced exponent placement ----------------------------------
    // Exponents for 1/x ~ sum_k w_k exp(-a_k x) on [1, R] should satisfy
    // a_k in [1/R, 1], i.e. a_k = exp(-t_k * logR) with t_k in (0,1).
    exponents.resize(nlap);
    for (int k = 0; k < nlap; ++k) {
        double t    = static_cast<double>(k + 1) / static_cast<double>(nlap + 1);
        exponents[k] = DD::ddExp(DD(-t * logRatio));
    }

    // When log(ratio)/nlap < 0.05 the exp columns are nearly identical and the
    // linear equioscillation system is severely ill-conditioned; the LU solve
    // produces huge cancelling weights whose aliased error appears small at the
    // 8*nlap sample points but is large everywhere else. Use original weak
    // regularisation and skip the linear Remez in that regime.
    const bool   useLinearRemez = (logRatio / nlap >= 0.05);
    const double lambdaFrac     = useLinearRemez ? 1e-4 : 1e-10;

    // ---- 2. Initial weights: Tikhonov LS -----------------------------------
    {
        std::vector<DD> G(nlap * nlap, DD(0.0));
        std::vector<DD> rhs(nlap, DD(0.0));
        std::vector<DD> phi(nlap);
        for (int j = 0; j < m; ++j) {
            double tj = static_cast<double>(j) / static_cast<double>(m - 1);
            DD xj     = DD::ddExp(DD(tj * logRatio));
            DD bj     = DD(1.0) / xj;
            for (int k = 0; k < nlap; ++k)
                phi[k] = DD::ddExp(-(exponents[k] * xj));
            for (int i = 0; i < nlap; ++i) {
                for (int k = 0; k < nlap; ++k)
                    G[i + k * nlap] += phi[i] * phi[k];
                rhs[i] += phi[i] * bj;
            }
        }
        double maxDiag = 0.0;
        for (int i = 0; i < nlap; ++i)
            maxDiag = std::max(maxDiag, G[i + i * nlap].hi);
        const double lambda = maxDiag * lambdaFrac;
        for (int i = 0; i < nlap; ++i)
            G[i + i * nlap] += DD(lambda);
        std::vector<int> piv(nlap);
        ddLuFactorize(G.data(), nlap, piv.data());
        ddLuSolve(G.data(), nlap, piv.data(), rhs.data());
        weights.resize(nlap);
        for (int k = 0; k < nlap; ++k)
            weights[k] = rhs[k];
    }

    if (!useLinearRemez) {
        errorAmplitude = computeInitialError(exponents.data(), weights.data(),
                                              nlap, ratio, m);
        return errorAmplitude.hi <= maxAcceptableError;
    }

    // ---- 3. Linear Remez: optimise weights only (exponents fixed) ----------
    // Chebyshev alternation theorem: nlap weights + ε → nlap+1 alternation pts.
    // Each iter: solve (nlap+1)×(nlap+1) equioscillation system, then update
    // the nlap-1 interior alternation points via Newton–Maehly.
    const int nEq = nlap + 1;
    const DD  rangeLo(1.0), rangeHi(ratio);

    std::vector<DD> eqPts(nEq);
    for (int i = 0; i < nEq; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(nEq - 1);
        eqPts[i] = DD::ddExp(DD(t * logRatio));
    }
    eqPts[0]     = rangeLo;
    eqPts[nEq-1] = rangeHi;

    std::vector<DD>  linA(nEq * nEq);
    std::vector<DD>  linRhs(nEq);
    std::vector<int> linPiv(nEq);

    for (int iter = 0; iter < 50; ++iter) {
        // Equioscillation system: Σ_k w_k exp(-a_k x_i) + sign_i*ε = 1/x_i
        // sign_0 = -1, alternating (matches evalEquioscillation convention).
        double sign = -1.0;
        for (int i = 0; i < nEq; ++i, sign = -sign) {
            for (int k = 0; k < nlap; ++k)
                linA[i + k * nEq] = DD::ddExp(-(exponents[k] * eqPts[i]));
            linA[i + nlap * nEq] = DD(sign);
            linRhs[i]            = DD(1.0) / eqPts[i];
        }
        ddLuFactorize(linA.data(), nEq, linPiv.data());
        ddLuSolve(linA.data(), nEq, linPiv.data(), linRhs.data());
        for (int k = 0; k < nlap; ++k)
            weights[k] = linRhs[k];
        errorAmplitude = DD::ddAbs(linRhs[nlap]);

        if (errorAmplitude.hi <= maxAcceptableError) break;
        if (nlap == 1) break;   // 0 interior extrema; nothing to update

        // Update interior alternation points via Newton–Maehly
        try {
            std::vector<DD> interior(nlap - 1);
            maehlySolver(interior.data(), nlap - 1,
                         rangeLo, rangeHi,
                         1000, 1e-10,
                         exponents.data(), weights.data(), nlap,
                         0.3, 1e-6, nullptr, 0, nullptr);
            for (int i = 0; i < nlap - 1; ++i)
                eqPts[i + 1] = interior[i];
        } catch (...) {
            break;
        }
    }

    // ---- 4. Dense error check ----------------------------------------------
    errorAmplitude = computeInitialError(exponents.data(), weights.data(),
                                          nlap, ratio, m);
    return errorAmplitude.hi <= maxAcceptableError;
}

/**
 * @brief Compute MINIMAX-optimal Laplace exponents and weights for 1/x on [ymin, ymax].
 *
 * Runs the Remez algorithm: alternating Newton-Maehly extremum search (maehlySolver) and
 * Newton-Raphson equioscillation solver (paraoptSolver) until convergence.  Uses 128-bit
 * double-double arithmetic throughout.
 *
 * @param nlap            Number of Laplace quadrature points; must satisfy 1 <= nlap <= 30.
 * @param ymin            Lower bound of the approximation interval (must be > 0).
 * @param ratio           Upper bound ratio ymax/ymin (must be > 1).
 * @param exponents       Initial exponents in the normalised [1, ratio] domain, length nlap.
 * @param weights         Initial weights   in the normalised [1, ratio] domain, length nlap.
 * @param errorAmplitude  Initial equioscillation error amplitude ε.
 * @param maxIter         Maximum Remez outer iterations (default 200).
 * @param toleranceMaehly Newton-Maehly convergence tolerance (default 1e-10).
 * @param toleranceNR     Newton-Raphson convergence tolerance (default 1e-15).
 * @param stepMax         Trust-region step cap for both inner solvers (default 0.3).
 * @param delta           Relative shift for initial extremum guess in maehlySolver (default 1e-6).
 * @param armijoConstant  Armijo constant for the inner line search (default 1e-4).
 * @param initExtrema     If non-null and non-empty, used as warm-start for the first
 *                        maehlySolver call only.  Length must be 2*nlap-1.
 * @param finalExtremaOut If non-null, filled with the converged interior extrema (length 2*nlap-1).
 *                        Used by tools/gen_table to capture the warm-start extrema stored in the tables.
 * @return minimax_cpppy::MinimaxResult with expon, weight, and errmax.
 */
static minimax_cpppy::MinimaxResult remezLoop(
    int nlap, double ymin, double ratio,
    std::vector<DD> exponents, std::vector<DD> weights, DD errorAmplitude,
    int maxIter, double toleranceMaehly, double toleranceNR,
    double stepMax, double delta, double armijoConstant,
    const std::vector<double>* initExtrema = nullptr,
    std::vector<double>* finalExtremaOut = nullptr,
    int verbose = 0, std::ostream* os = nullptr)
{
    const int numPoints = 2 * nlap + 1;
    std::vector<DD> interiorExtrema(numPoints - 2), allExtrema(numPoints);
    DD rangeLo(1.0), rangeHi(ratio);

    const double* maehlyHint = (initExtrema && !initExtrema->empty())
                               ? initExtrema->data() : nullptr;

    if (verbose >= 3 && os) {
        *os << "[Remez]\n"
            << fmtHeaderCell("iter", 4) << "  " << fmtHeaderCell("errmax", 12)
            << "  " << fmtHeaderCell("NR_iters", 8) << "\n"
            << ruleCell(4) << "  " << ruleCell(12) << "  " << ruleCell(8) << "\n";
    }
    for (int iter = 0; iter < maxIter; ++iter) {
        maehlySolver(interiorExtrema.data(), numPoints - 2, rangeLo, rangeHi,
                     std::max(maxIter * 5, 1000), toleranceMaehly,
                     exponents.data(), weights.data(),
                     nlap, stepMax, delta, maehlyHint, verbose, os);
        maehlyHint = nullptr;   // hint used only on first iteration

        allExtrema[0]             = rangeLo;
        allExtrema[numPoints - 1] = rangeHi;
        for (int i = 0; i < numPoints - 2; ++i) { allExtrema[i + 1] = interiorExtrema[i]; }

        int numNRIters = paraoptSolver(exponents.data(), weights.data(), errorAmplitude,
                                       allExtrema.data(), numPoints,
                                       maxIter, toleranceNR, stepMax, armijoConstant, nlap,
                                       verbose, os);
        if (verbose >= 3 && os) {
            *os << fmtCellInt(iter, 4) << "  " << fmtCell(std::abs(errorAmplitude.hi), 12, 4)
                << "  " << fmtCellInt(numNRIters, 8) << "\n";
        }
        if (numNRIters == 1) { break; }
    }

    if (finalExtremaOut != nullptr) {
        finalExtremaOut->resize(numPoints - 2);
        for (int i = 0; i < numPoints - 2; ++i) {
            (*finalExtremaOut)[i] = interiorExtrema[i].hi;
        }
    }

    minimax_cpppy::MinimaxResult result;
    result.expon.resize(nlap);
    result.weight.resize(nlap);
    for (int k = 0; k < nlap; ++k) {
        result.expon[k]  = (exponents[k] / ymin).hi;
        result.weight[k] = (weights[k]   / ymin).hi;
    }
    result.errmax = std::abs(errorAmplitude.hi);
    return result;
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_REMEZ_HPP
