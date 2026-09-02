/**
 * @file paraopt.hpp
 * @brief Newton-Raphson equioscillation solver for the MINIMAX Remez algorithm.
 *
 * Contains the equioscillation residual/Jacobian evaluator, the Armijo backtracking
 * line search, and the Newton-Raphson solver (paraoptSolver) that updates the
 * exponents, weights, and error amplitude to satisfy equioscillation conditions.
 */
#pragma once
#ifndef MINIMAX_CPPPY_PARAOPT_HPP
#define MINIMAX_CPPPY_PARAOPT_HPP

#include <cmath>
#include <ostream>
#include <stdexcept>
#include <vector>
#include "maehly.hpp"
#include "log_pretty.hpp"

namespace minimax_cpppy {
namespace detail {

/**
 * @brief Evaluate the equioscillation residual vector (computeMode=1) or its Jacobian (computeMode=2).
 *
 * The system state is packed as stateVec = [exponents[0..nlap-1], weights[0..nlap-1], ε].
 * residual_i = ε·sign_i − 1/x_i + Σ_k w_k exp(−a_k x_i),  sign_i = (−1)^i.
 * The Jacobian is column-major: jacobian[row + col*systemDim].
 *
 * @param residual    Output residual vector (length systemDim); written when computeMode==1.
 * @param jacobian    Output Jacobian matrix (systemDim × systemDim, col-major); written when computeMode==2.
 * @param computeMode 1 = compute residual only; 2 = compute jacobian only.
 * @param nlap        Number of Laplace points.
 * @param extremaPoints Equioscillation evaluation points, length systemDim.
 * @param stateVec    Packed [exponents, weights, ε] state vector, length systemDim.
 * @param systemDim   System dimension = 2*nlap + 1.
 */
static void evalEquioscillation(DD* residual, DD* jacobian, int computeMode,
                                 int nlap, const DD* extremaPoints,
                                 const DD* stateVec, int systemDim) noexcept
{
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    ++minimax_cpppy::debug::eval_equioscillation_call_count;
    if (!extremaPoints || !stateVec) {
        throw std::invalid_argument("evalEquioscillation: null pointer argument");
    }
    if (nlap <= 0 || systemDim <= 0) {
        throw std::invalid_argument("evalEquioscillation: non-positive nlap or systemDim");
    }
    if (computeMode == 1 && !residual) {
        throw std::invalid_argument("evalEquioscillation: null residual for computeMode==1");
    }
    if (computeMode == 2 && !jacobian) {
        throw std::invalid_argument("evalEquioscillation: null jacobian for computeMode==2");
    }
#endif
    if (computeMode == 1) {
        double sign = -1.0;
        for (int i = 0; i < systemDim; ++i, sign = -sign) {
            // residual[i] = ε*sign - 1/x_i + Σ w_k exp(-a_k x_i)
            // Fortran: temp1 = ε*sign; temp2 = temp1*x - 1; residual = temp2/x
            DD temp      = (stateVec[systemDim-1] * sign) * extremaPoints[i] - DD(1.0);
            residual[i]  = temp / extremaPoints[i];
            for (int k = 0; k < nlap; ++k) {
                DD expTerm   = DD::ddExp(-(stateVec[k] * extremaPoints[i]));
                residual[i] += stateVec[nlap + k] * expTerm;
            }
        }
    } else {  // computeMode == 2: Jacobian only, residual unchanged
        double sign = -1.0;
        for (int i = 0; i < systemDim; ++i, sign = -sign) {
            for (int k = 0; k < nlap; ++k) {
                DD expTerm         = DD::ddExp(-(stateVec[k] * extremaPoints[i]));
                DD weightedExpTerm = stateVec[nlap+k] * expTerm;
                jacobian[i + k*systemDim]        = -(weightedExpTerm * extremaPoints[i]);
                jacobian[i + (nlap+k)*systemDim] = expTerm;
            }
            jacobian[i + (systemDim-1)*systemDim] = DD(sign);
        }
    }
}

/**
 * @brief Backtracking line search for the Newton step in the equioscillation NR system.
 *
 * Updates stateVec and residual to the accepted trial point.
 * Accepts if the Armijo condition f(stateVec) <= refHalfFuncNormSq + armijoConstant * stepLength * slope is met.
 *
 * @param stateVec          In/out packed state vector; updated to accepted point.
 * @param residual          In/out residual vector; updated to F(stateVec) at accepted point.
 * @param prevStateVec      State vector at the start of the NR step (not modified).
 * @param newtonStep        Newton descent direction, length systemDim.
 * @param gradient          Gradient of 0.5*||F||² at prevStateVec, length systemDim.
 * @param refHalfFuncNormSq 0.5*||F(prevStateVec)||² (Armijo reference value).
 * @param armijoConstant    Armijo decrease constant (typically 1e-4).
 * @param systemDim         System dimension.
 * @param nlap              Number of Laplace points.
 * @param extremaPoints     Equioscillation points, length systemDim.
 * @param maxIter           Maximum backtracking iterations.
 */
static void linesearch(DD* stateVec, DD* residual,
                        const DD* prevStateVec, const DD* newtonStep, const DD* gradient,
                        double refHalfFuncNormSq, double armijoConstant,
                        int systemDim, int nlap, const DD* extremaPoints, int maxIter) noexcept
{
    double slope = 0.0;
    for (int i = 0; i < systemDim; ++i) { slope += gradient[i].hi * newtonStep[i].hi; }

    const double minStepLength = 1e-15;
    double stepLength = 1.0, prevStepLength = 0.0, prevHalfFuncNormSq = 0.0;

    for (int it = 0; it < maxIter; ++it) {
        for (int i = 0; i < systemDim; ++i) {
            stateVec[i] = prevStateVec[i] + DD(stepLength) * newtonStep[i];
        }
        evalEquioscillation(residual, nullptr, 1, nlap, extremaPoints, stateVec, systemDim);

        double halfFuncNormSq = 0.0;
        for (int i = 0; i < systemDim; ++i) { halfFuncNormSq += residual[i].hi * residual[i].hi; }
        halfFuncNormSq *= 0.5;

        if (stepLength <= minStepLength) {
            for (int i = 0; i < systemDim; ++i) { stateVec[i] = prevStateVec[i]; }
            return;
        }
        if (halfFuncNormSq <= refHalfFuncNormSq + armijoConstant * stepLength * slope) { return; }

        // next step length by interpolation
        double nextStepLength;
        if (it == 0) {
            double a  = halfFuncNormSq - refHalfFuncNormSq - slope;
            nextStepLength = (std::abs(a) < 1e-300) ? 0.5 : -slope / (2.0 * a);
        } else {
            double rhs1 = halfFuncNormSq     - refHalfFuncNormSq - stepLength     * slope;
            double rhs2 = prevHalfFuncNormSq - refHalfFuncNormSq - prevStepLength * slope;
            double a    = (rhs1/(stepLength*stepLength) - rhs2/(prevStepLength*prevStepLength))
                          / (stepLength - prevStepLength);
            double b    = (-prevStepLength*rhs1/(stepLength*stepLength)
                           + stepLength*rhs2/(prevStepLength*prevStepLength))
                          / (stepLength - prevStepLength);
            if (std::abs(a) < 1e-300) {
                nextStepLength = -slope / (2.0 * b);
            } else {
                double disc    = b*b - 3.0*a*slope;
                nextStepLength = (-b + std::sqrt(std::max(disc, 0.0))) / (3.0 * a);
            }
            if (nextStepLength > 0.5 * stepLength) { nextStepLength = 0.5 * stepLength; }
        }
        prevStepLength     = stepLength;
        prevHalfFuncNormSq = halfFuncNormSq;
        stepLength         = std::max(nextStepLength, 0.1 * stepLength);
    }
}

/**
 * @brief Newton-Raphson solver for the equioscillation system (Remez parameter update).
 *
 * Iteratively solves the nonlinear equioscillation conditions to find exponents,
 * weights, and error magnitude such that the Chebyshev equioscillation holds.
 *
 * @param exponents     In/out exponents a_k, length nlap; updated on return.
 * @param weights       In/out weights w_k, length nlap; updated on return.
 * @param errorAmplitude In/out equioscillation error amplitude ε; updated on return.
 * @param extremaPoints  Fixed evaluation points (extrema), length numPoints.
 * @param numPoints      Number of extremum points (= 2*nlap+1).
 * @param maxIter        Maximum NR iterations.
 * @param tolerance      Convergence tolerance on ||δx||.
 * @param stepMax        Maximum step size (trust-region cap).
 * @param armijoConstant Armijo constant for the inner line search.
 * @param nlap           Number of Laplace points.
 * @return Number of NR iterations taken; 1 means converged on the first step.
 */
static int paraoptSolver(DD* exponents, DD* weights, DD& errorAmplitude,
                          const DD* extremaPoints, int numPoints,
                          int maxIter, double tolerance, double stepMax,
                          double armijoConstant, int nlap,
                          int verbose = 0, std::ostream* os = nullptr)
{
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (!exponents || !weights || !extremaPoints) {
        throw std::invalid_argument("paraoptSolver: null pointer argument");
    }
    if (nlap <= 0 || numPoints <= 0) {
        throw std::invalid_argument("paraoptSolver: non-positive nlap or numPoints");
    }
#endif
    const int systemDim = numPoints;
    std::vector<DD> stateVec(systemDim), prevStateVec(systemDim),
                    newtonStep(systemDim), gradient(systemDim), residual(systemDim);
    std::vector<DD> jacobian(systemDim * systemDim);
    std::vector<int> pivotIndices(systemDim);

    for (int k = 0; k < nlap; ++k) {
        stateVec[k]        = exponents[k];
        stateVec[nlap + k] = weights[k];
    }
    stateVec[systemDim - 1] = errorAmplitude;

    // initial residual
    evalEquioscillation(residual.data(), jacobian.data(), 1, nlap,
                        extremaPoints, stateVec.data(), systemDim);
    double halfFuncNormSq = 0.0;
    for (int i = 0; i < systemDim; ++i) { halfFuncNormSq += residual[i].hi * residual[i].hi; }
    halfFuncNormSq *= 0.5;

    if (verbose >= 3 && os) {
        *os << "  [ParaOpt]\n"
            << "  " << fmtHeaderCell("iter", 4) << "  " << fmtHeaderCell("stepNorm", 12)
            << "  " << fmtHeaderCell("||F||^2", 12) << "\n"
            << "  " << ruleCell(4) << "  " << ruleCell(12) << "  " << ruleCell(12) << "\n";
    }
    int iter = 1;
    for (; iter <= maxIter; ++iter) {
        // Jacobian at stateVec (residual retains F(stateVec) from previous iteration / init)
        evalEquioscillation(residual.data(), jacobian.data(), 2, nlap,
                            extremaPoints, stateVec.data(), systemDim);

        // gradient of 0.5*||F||^2 = J^T * F
        ddGemvTransposed(systemDim, systemDim, DD(1.0), jacobian.data(), systemDim,
                         residual.data(), DD(0.0), gradient.data());

        // Newton RHS: newtonStep = -F
        for (int i = 0; i < systemDim; ++i) { newtonStep[i] = -residual[i]; }

        // LU factorize J (overwrites jacobian), solve J*newtonStep = -F
        ddLuFactorize(jacobian.data(), systemDim, pivotIndices.data());
        ddLuSolve(jacobian.data(), systemDim, pivotIndices.data(), newtonStep.data());

        // scale step if it exceeds the trust-region cap
        DD stepNorm = ddNorm2(systemDim, newtonStep.data());
        if (stepNorm > DD(stepMax)) {
            ddScale(systemDim, DD(stepMax) / stepNorm, newtonStep.data());
        }

        // line search: updates stateVec and residual
        ddCopy(systemDim, stateVec.data(), prevStateVec.data());
        linesearch(stateVec.data(), residual.data(),
                   prevStateVec.data(), newtonStep.data(), gradient.data(),
                   halfFuncNormSq, armijoConstant, systemDim, nlap, extremaPoints, maxIter);
        halfFuncNormSq = 0.0;
        for (int i = 0; i < systemDim; ++i) { halfFuncNormSq += residual[i].hi * residual[i].hi; }
        halfFuncNormSq *= 0.5;

        if (verbose >= 3 && os) {
            *os << "  " << fmtCellInt(iter, 4) << "  " << fmtCell(stepNorm.hi, 12, 4)
                << "  " << fmtCell(2.0 * halfFuncNormSq, 12, 4) << "\n";
        }
        if (stepNorm <= DD(tolerance)) { break; }
    }

    for (int k = 0; k < nlap; ++k) {
        exponents[k] = stateVec[k];
        weights[k]   = stateVec[nlap + k];
    }
    errorAmplitude = stateVec[systemDim - 1];
    return iter;
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_PARAOPT_HPP
