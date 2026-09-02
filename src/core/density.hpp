/**
 * @file density.hpp
 * @brief Pair-denominator density estimation for biased Remez quadrature.
 *
 * Builds a smooth, normalised weighting density p(t) on the log-denominator axis
 * t = ln(Δ/Δ_min) ∈ [0, t_max], where Δ = 2(ε_a − ε_i) is the Møller-Plesset
 * pair denominator.  The density is estimated via KDE with Gaussian smoothing in
 * Δ-space using FFT convolution, then transformed to log-space, optionally weighted
 * by an energy-importance decay e^{−α t}, subjected to an entropy-adaptive floor,
 * renormalised, and stored in a quintic Hermite spline for C²-smooth evaluation.
 *
 * The resulting @ref DensityArrays is consumed by the biased Remez solver in
 * biasing.cpp to concentrate quadrature accuracy where the pair-denominator density
 * is highest.
 */
#pragma once
#include "ifft.hpp"
#include <algorithm>
#include <complex>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax_cpppy {
namespace detail {

/**
 * @brief Output of buildDensityArrays(): knot data for the C²-smooth density spline.
 *
 * All arrays have the same length (the number of log-grid knots, @c numLogGridPoints
 * passed to buildDensityArrays()).  The quintic Hermite spline defined by these arrays
 * maps t ∈ [0, logRatioMax] to a normalised weighting density and its first two
 * derivatives.
 */
struct DensityArrays {
    std::vector<double> logGridKnots;         ///< Knot positions t_j on the log-denominator axis.
    std::vector<double> weightKnots;          ///< Normalised density w(t_j) at each knot.
    std::vector<double> weightDerivKnots;     ///< First derivative w'(t_j) at each knot.
    std::vector<double> weightSecondDerivKnots; ///< Second derivative w''(t_j) at each knot.
    double denominatorMin; ///< Minimum pair denominator Δ_min = 2(ε_v_min − ε_o_max) [Ha].
    double denominatorMax; ///< Maximum pair denominator Δ_max = 2(ε_v_max − ε_o_min) [Ha].
    double ratio;          ///< Denominator ratio Δ_max / Δ_min (= exp(logRatioMax)).
    double logRatioMax;    ///< Upper limit of the log-denominator axis: ln(Δ_max / Δ_min).
};

// ── cubic spline helpers ───────────────────────────────────────────────────

/**
 * @brief Natural cubic spline interpolant with C² evaluation.
 *
 * Builds a piecewise-cubic spline through a set of knots with natural boundary
 * conditions (second derivatives zero at the endpoints) using the Thomas algorithm
 * for the tridiagonal system.  Provides simultaneous evaluation of the spline
 * value and its first two derivatives at any query point.
 *
 * Used internally by buildDensityArrays() to interpolate the KDE density
 * p_Δ(Δ) from the linear FFT grid onto the log-denominator t-grid.
 */
struct CubicSpline {
    std::vector<double> knotPositions_; ///< Knot abscissae x_i.
    std::vector<double> knotValues_;    ///< Knot ordinates y_i.
    std::vector<double> knotSecondDerivatives_; ///< Natural spline second derivatives M_i at knots.

    /**
     * @brief Build the natural cubic spline through the given knot data.
     *
     * Solves the symmetric tridiagonal second-derivative system using the Thomas
     * (forward-sweep / back-substitution) algorithm.  Sets M_0 = M_n = 0 (natural
     * boundary conditions).  If fewer than three knots are provided, the spline
     * degenerates to linear interpolation (all M_i = 0).
     *
     * @param x  Strictly increasing knot abscissae, length ≥ 2.
     * @param y  Knot ordinates corresponding to @p x, same length.
     */
    void build(const std::vector<double>& x, const std::vector<double>& y) {
        const int numIntervals = static_cast<int>(x.size()) - 1;
        knotPositions_ = x;  knotValues_ = y;  knotSecondDerivatives_.assign(x.size(), 0.0);
        if (numIntervals < 2) return;

        std::vector<double> intervalWidths(numIntervals),
                            subdiagonal(numIntervals-1),
                            mainDiagonal(numIntervals-1),
                            rightHandSide(numIntervals-1),
                            thomasMultiplier(numIntervals-1),
                            thomasValue(numIntervals-1);
        for (int i = 0; i < numIntervals; ++i) intervalWidths[i] = x[i+1] - x[i];
        for (int i = 0; i < numIntervals-1; ++i) {
            mainDiagonal[i] = 2.0 * (intervalWidths[i] + intervalWidths[i+1]);
            subdiagonal[i]  = intervalWidths[i+1];
            rightHandSide[i] = 6.0 * ((y[i+2]-y[i+1])/intervalWidths[i+1]
                                       - (y[i+1]-y[i])/intervalWidths[i]);
        }
        // Thomas forward sweep
        thomasMultiplier[0] = subdiagonal[0] / mainDiagonal[0];
        thomasValue[0]      = rightHandSide[0] / mainDiagonal[0];
        for (int i = 1; i < numIntervals-1; ++i) {
            double denom        = mainDiagonal[i] - intervalWidths[i] * thomasMultiplier[i-1];
            thomasMultiplier[i] = subdiagonal[i] / denom;
            thomasValue[i]      = (rightHandSide[i] - intervalWidths[i] * thomasValue[i-1]) / denom;
        }
        // Back-substitute (knotSecondDerivatives_[0] = knotSecondDerivatives_[n] = 0 by natural BC)
        knotSecondDerivatives_[numIntervals-1] = thomasValue[numIntervals-2];
        for (int i = numIntervals-2; i >= 1; --i)
            knotSecondDerivatives_[i] = thomasValue[i-1] - thomasMultiplier[i-1] * knotSecondDerivatives_[i+1];
    }

    /**
     * @brief Evaluate the spline and its first two derivatives at a query point.
     *
     * Uses binary search to locate the enclosing interval, then evaluates the
     * piecewise-cubic formula directly.  The query point is clamped to the first
     * or last interval if it lies outside the knot range.
     *
     * @param t    Query abscissa.
     * @param p    Output: spline value at @p t.
     * @param dp   Output: first derivative at @p t.
     * @param d2p  Output: second derivative at @p t.
     */
    void eval(double t, double& p, double& dp, double& d2p) const {
        const int numIntervals = static_cast<int>(knotPositions_.size()) - 1;
        int j = static_cast<int>(
            std::lower_bound(knotPositions_.begin(), knotPositions_.end(), t)
            - knotPositions_.begin()) - 1;
        j = std::max(0, std::min(j, numIntervals-1));
        double intervalWidth   = knotPositions_[j+1] - knotPositions_[j];
        double localOffset     = t - knotPositions_[j];
        double leftSecondDeriv  = knotSecondDerivatives_[j];
        double rightSecondDeriv = knotSecondDerivatives_[j+1];
        double leftValue        = knotValues_[j];
        double rightValue       = knotValues_[j+1];
        double linearCoeff = (rightValue - leftValue) / intervalWidth
                             - intervalWidth * (2.0*leftSecondDeriv + rightSecondDeriv) / 6.0;
        p   = leftValue + linearCoeff*localOffset
              + 0.5*leftSecondDeriv*localOffset*localOffset
              + (rightSecondDeriv-leftSecondDeriv)/(6.0*intervalWidth)*localOffset*localOffset*localOffset;
        dp  = linearCoeff + leftSecondDeriv*localOffset
              + 0.5*(rightSecondDeriv-leftSecondDeriv)/intervalWidth*localOffset*localOffset;
        d2p = leftSecondDeriv + (rightSecondDeriv-leftSecondDeriv)/intervalWidth*localOffset;
    }
};

// The C²-smooth quintic Hermite spline that consumes these knots is evaluated
// at run time by DenominatorDensity::evalW (include/minimax_cpppy/biasing.hpp).

// ── main builder ──────────────────────────────────────────────────────────

/**
 * @brief Build the log-space pair-denominator density for biased Remez quadrature.
 *
 * Estimates the distribution of MP2 pair denominators Δ = 2(ε_a − ε_i) via
 * Gaussian-smoothed KDE in Δ-space using FFT convolution (cost linear in
 * @p fftSize × (@p numOccupied + @p numVirtual)).  The result is resampled to a
 * log-denominator grid t = ln(Δ/Δ_min), optionally down-weighted by an energy-
 * importance factor e^{−α t}, subjected to an entropy-adaptive floor, renormalised,
 * and packaged as a quintic Hermite spline stored in the returned @ref DensityArrays.
 *
 * ### Bandwidth parameter
 * The Gaussian kernel standard deviation is σ_Δ = @p bandwidth × Δ_max / 4.
 * At @p bandwidth = 1.0, σ ≈ 25 % of the denominator range, giving a nearly flat
 * density that acts as a gentle perturbation of the unbiased Remez and consistently
 * improves SOS-MP2 energies.  Values below 0.3 produce a spiky density that
 * concentrates accuracy at moderate Δ while relaxing it near the HOMO-LUMO gap,
 * which typically degrades accuracy.  The recommended range is 0.5–2.0.
 *
 * @param occ             Occupied orbital energies [Ha], length @p numOccupied.
 * @param numOccupied     Number of occupied orbitals.
 * @param virt            Virtual orbital energies [Ha], length @p numVirtual.
 * @param numVirtual      Number of virtual orbitals.
 * @param bandwidth       KDE bandwidth (see above); 1.0 is a good default.
 * @param fftSize         Number of FFT grid points; should be a power of 2 for efficiency.
 * @param numLogGridPoints Number of knots on the log-denominator t-grid.
 * @param floorFraction   Floor as a fraction of the density peak; prevents the spline
 *                        weight from dropping to zero (recommended: 0.05–0.2).
 * @param floorFractionMax Upper limit of the entropy-adaptive floor fraction.
 *                        When @p floorFractionMax > @p floorFraction, the floor
 *                        fraction is interpolated between @p floorFraction (uniform
 *                        density, high Shannon entropy) and @p floorFractionMax
 *                        (narrow density, low entropy).  Set ≤ @p floorFraction to
 *                        disable adaptation (default: −1.0).
 * @param n_exc          Excitation order: 1 for singles, 2 for doubles, etc.
 *                        Bounds are n_exc × D_single_{min,max}; the characteristic
 *                        function is φ_single^n_exc / N^n_exc where N = n_occ·n_virt.
 *                        Must be ≥ 1.  Default: 2.
 * @return DensityArrays containing the quintic Hermite spline knots and the
 *         physical denominator bounds.
 * @throws std::invalid_argument if the occupied and virtual energy ranges overlap
 *         (Δ_min ≤ 0).
 * @throws std::invalid_argument if n_exc < 1.
 * @throws std::runtime_error if the KDE density integrates to zero after clipping.
 */
inline DensityArrays buildDensityArrays(
    const double* occ,  int numOccupied,
    const double* virt, int numVirtual,
    double bandwidth,
    int fftSize,
    int numLogGridPoints,
    double floorFraction,
    double floorFractionMax = -1.0,
    int n_exc = 2)
{
    if (n_exc < 1)
        throw std::invalid_argument(
            "buildDensityArrays: n_exc must be >= 1, got " + std::to_string(n_exc));
    using Cx = std::complex<double>;
    static constexpr double pi = 3.14159265358979323846264338327950288;

    // Step 1 — sort orbitals, compute n_exc-fold denominator bounds.
    // D_pair = D_ia1 + … + D_ia_n_exc; bounds are n_exc × single-excitation bounds.
    std::vector<double> occSorted(occ, occ + numOccupied);
    std::vector<double> virtSorted(virt, virt + numVirtual);
    std::stable_sort(occSorted.begin(), occSorted.end());
    std::stable_sort(virtSorted.begin(), virtSorted.end());

    double D_single_min = virtSorted.front() - occSorted.back();
    double D_single_max = virtSorted.back()  - occSorted.front();
    double denominatorMin         = n_exc * D_single_min;
    double denominatorMaxPhysical = n_exc * D_single_max;

    if (denominatorMin <= 0.0)
        throw std::invalid_argument(
            "buildDensityArrays: delta_min <= 0 (occ and virt overlap)");

    double physicalLogRatio = std::log(denominatorMaxPhysical / denominatorMin);

    // Degenerate: single orbital pair → ratio ≈ 1, uniform density.
    // H0+H3 = 1 identically for quintic Hermite, so a flat 2-knot spline
    // evaluates to (1, 0, 0) for any t, including extrapolation.
    // Use a minimum logRatioMax of 0.01 (the branch threshold) so that ratio > 1
    // and delta_max > delta_min are always guaranteed.
    if (physicalLogRatio < 0.01) {
        double degenerateLogRatio = std::max(physicalLogRatio, 0.01);
        DensityArrays out;
        out.logGridKnots          = {0.0, degenerateLogRatio};
        out.weightKnots           = {1.0, 1.0};
        out.weightDerivKnots      = {0.0, 0.0};
        out.weightSecondDerivKnots = {0.0, 0.0};
        out.denominatorMin = denominatorMin;
        out.denominatorMax = denominatorMin * std::exp(degenerateLogRatio);
        out.ratio          = std::exp(degenerateLogRatio);
        out.logRatioMax    = degenerateLogRatio;
        return out;
    }

    // σ_Δ = bandwidth · Δ_max / 4: quarter-range scaling.
    // The biased minimax works best when σ is large enough to give a nearly flat
    // density (≈25% of the pair denominator range) so that biasing is a gentle
    // perturbation of the unbiased solution.  The old formula
    // (bandwidth · ln(ratio) · Δ_min) gave σ ≈ 8% of range at bandwidth=1.0 for
    // typical quantum-chemistry systems, which produced a spiky density that
    // concentrated accuracy in the wrong region and degraded SOS-MP2 accuracy.
    double kernelStdDev = bandwidth * denominatorMaxPhysical / 4.0;
    // Extend FFT grid to cover [0, 2·physicalLogRatio] in t-space
    // (denominatorMin·ratio² in Δ-space).  The buffer region [physicalLogRatio, 2·physicalLogRatio]
    // is discarded after iFFT when we sample only the physical t-grid [0, physicalLogRatio] in Step 8.
    double denominatorFftTop  = denominatorMaxPhysical * (denominatorMaxPhysical / denominatorMin);
    double denominatorFftMax  = denominatorFftTop + 5.0 * kernelStdDev;
    double fftGridSpan        = denominatorFftMax - denominatorMin;
    double denominatorGridSpacing = fftGridSpan / fftSize;

    // Step 2 — centred Fourier frequency grid
    double frequencySpacing = 2.0 * pi / (fftSize * denominatorGridSpacing);
    std::vector<double> frequencyGrid(fftSize);
    for (int n = 0; n < fftSize; ++n)
        frequencyGrid[n] = (n - fftSize / 2) * frequencySpacing;

    // Steps 3+4 — denominator characteristic function via φ_single^n_exc.
    // φ_Δ(k) = Gaussian(σ,k) · [φ_virt(k) · φ_occ(k)]^n_exc / N^n_exc
    // where φ_virt(k) = Σ_a exp(-ik·ε_a), φ_occ(k) = Σ_i exp(+ik·ε_i),
    // N = n_occ · n_virt.  Cost: O(n_fft · (n_occ + n_virt)) — linear in system size.
    double N_single = static_cast<double>(numOccupied) * static_cast<double>(numVirtual);
    double N_n_exc  = std::pow(N_single, static_cast<double>(n_exc));

    double sigmaSquared = kernelStdDev * kernelStdDev;
    std::vector<Cx> denominatorCharacteristicFunc(fftSize);
    for (int n = 0; n < fftSize; ++n) {
        double k              = frequencyGrid[n];
        double gaussianFactor = std::exp(-sigmaSquared * k * k);

        Cx phi_virt(0.0, 0.0);
        for (int a = 0; a < numVirtual; ++a)
            phi_virt += std::exp(Cx(0.0, -k * virtSorted[a]));

        Cx phi_occ(0.0, 0.0);
        for (int i = 0; i < numOccupied; ++i)
            phi_occ += std::exp(Cx(0.0, +k * occSorted[i]));

        Cx phi_single = phi_virt * phi_occ;
        Cx phi_n(1.0, 0.0);
        for (int e = 0; e < n_exc; ++e)
            phi_n *= phi_single;

        denominatorCharacteristicFunc[n] = gaussianFactor * phi_n / N_n_exc;
    }

    // Step 5 — IFFT to recover p_Δ on linear grid
    // Phase shift for Δ_min offset + fftshift (move DC from fftSize/2 to 0)
    std::vector<Cx> ifftSpectrum(fftSize);
    for (int n = 0; n < fftSize; ++n) {
        double frequencyValue  = frequencyGrid[n];
        Cx     phaseShiftedValue = denominatorCharacteristicFunc[n]
                                   * std::exp(Cx(0, frequencyValue * denominatorMin));
        int    destinationIndex  = (n + fftSize / 2) % fftSize;
        ifftSpectrum[destinationIndex] = phaseShiftedValue;
    }
    // ifft (1/N normalization built in)
    std::vector<double> densityLinearGridRaw = ifft<double>(ifftSpectrum);
    // Density units: multiply by 1/d_delta
    double densityNormalizationFactor = 1.0 / denominatorGridSpacing;
    std::vector<double> densityOnLinearGrid(fftSize);
    for (int m = 0; m < fftSize; ++m)
        densityOnLinearGrid[m] = densityLinearGridRaw[m] * densityNormalizationFactor;

    // Step 6 — clip negatives and renormalize
    double densityIntegral = 0.0;
    for (int m = 0; m < fftSize; ++m) {
        if (densityOnLinearGrid[m] < 0.0) densityOnLinearGrid[m] = 0.0;
        densityIntegral += densityOnLinearGrid[m] * denominatorGridSpacing;
    }
    if (densityIntegral <= 0.0)
        throw std::runtime_error("buildDensityArrays: p_delta norm is zero");
    for (int m = 0; m < fftSize; ++m)
        densityOnLinearGrid[m] /= densityIntegral;

    // Δ grid: Δ_m = Δ_min + m * denominatorGridSpacing
    std::vector<double> denominatorLinearGrid(fftSize);
    for (int m = 0; m < fftSize; ++m)
        denominatorLinearGrid[m] = denominatorMin + m * denominatorGridSpacing;

    // Step 7 — natural cubic spline on (denominatorLinearGrid, densityOnLinearGrid)
    CubicSpline deltaSpline;
    deltaSpline.build(denominatorLinearGrid, densityOnLinearGrid);

    // Step 8 — resample to log t-grid (physical domain only; FFT buffer region discarded)
    std::vector<double> logGridPoints(numLogGridPoints), denominatorAtLogGrid(numLogGridPoints);
    for (int j = 0; j < numLogGridPoints; ++j) {
        logGridPoints[j]        = physicalLogRatio * j / (numLogGridPoints - 1);
        denominatorAtLogGrid[j] = denominatorMin * std::exp(logGridPoints[j]);
    }

    // Evaluate p_Δ, p_Δ', p_Δ'' at each Δ_j via cubic spline
    std::vector<double> densityAtLogGrid(numLogGridPoints),
                        densityDerivAtLogGrid(numLogGridPoints),
                        densitySecondDerivAtLogGrid(numLogGridPoints);
    for (int j = 0; j < numLogGridPoints; ++j)
        deltaSpline.eval(denominatorAtLogGrid[j],
                         densityAtLogGrid[j],
                         densityDerivAtLogGrid[j],
                         densitySecondDerivAtLogGrid[j]);

    // Step 9 — Jacobian transform to t-space: w(t) = p_Δ(Δ)·Δ where Δ = Δ_min·e^t.
    // Chain-rule derivatives (dΔ/dt = Δ):
    //   w'  = p_Δ'·Δ² + p_Δ·Δ
    //   w'' = p_Δ''·Δ³ + 3·p_Δ'·Δ² + p_Δ·Δ
    std::vector<double> logSpaceDensity(numLogGridPoints),
                        logSpaceDensityDeriv(numLogGridPoints),
                        logSpaceDensitySecondDeriv(numLogGridPoints);
    for (int j = 0; j < numLogGridPoints; ++j) {
        double D = denominatorAtLogGrid[j];
        logSpaceDensity[j]            = densityAtLogGrid[j] * D;
        logSpaceDensityDeriv[j]       = D   * densityAtLogGrid[j]
                                      + D*D * densityDerivAtLogGrid[j];
        logSpaceDensitySecondDeriv[j] = D     * densityAtLogGrid[j]
                                      + 3.0*D*D * densityDerivAtLogGrid[j]
                                      + D*D*D   * densitySecondDerivAtLogGrid[j];
    }

    // Step 9d — entropy-adaptive floor.
    // Narrow distributions (low Shannon entropy, like N2) get a higher floor
    // so equioscillation points spread over the full [0, physicalLogRatio] range.
    // floorFractionMax <= floorFraction disables adaptation.
    double effectiveFloorFraction = floorFraction;
    if (floorFractionMax > floorFraction) {
        // Differential entropy of logSpaceDensity on [0, physicalLogRatio] (trapezoidal, nats).
        double densityNormalization = 0.0;
        for (int j = 0; j < numLogGridPoints - 1; ++j)
            densityNormalization += 0.5 * (logSpaceDensity[j] + logSpaceDensity[j+1])
                                    * (logGridPoints[j+1] - logGridPoints[j]);
        double shannonEntropy = 0.0;
        if (densityNormalization > 0.0) {
            for (int j = 0; j < numLogGridPoints - 1; ++j) {
                double dt = logGridPoints[j+1] - logGridPoints[j];
                for (int s = 0; s < 2; ++s) {
                    double p = (s == 0 ? logSpaceDensity[j] : logSpaceDensity[j+1])
                               / densityNormalization;
                    if (p > 1e-300) shannonEntropy -= p * std::log(p) * dt * 0.5;
                }
            }
        }
        double uniformReferenceEntropy = std::log(physicalLogRatio > 0.0 ? physicalLogRatio : 1.0);
        double uniformityRatio = (uniformReferenceEntropy > 0.0)
            ? std::max(0.0, std::min(1.0, shannonEntropy / uniformReferenceEntropy))
            : 1.0;
        // uniformityRatio→0 (narrow) → floorFractionMax;  uniformityRatio→1 (uniform) → floorFraction
        effectiveFloorFraction = floorFractionMax
                                 + uniformityRatio * (floorFraction - floorFractionMax);
    }

    // Step 10 — apply density floor to prevent the weight function from collapsing to zero.
    double densityPeak = *std::max_element(logSpaceDensity.begin(), logSpaceDensity.end());
    double floorValue  = effectiveFloorFraction * densityPeak;
    std::vector<double> effectiveWeight(numLogGridPoints),
                        effectiveWeightDeriv(numLogGridPoints),
                        effectiveWeightSecondDeriv(numLogGridPoints);
    for (int j = 0; j < numLogGridPoints; ++j) {
        if (logSpaceDensity[j] >= floorValue) {
            effectiveWeight[j]            = logSpaceDensity[j];
            effectiveWeightDeriv[j]       = logSpaceDensityDeriv[j];
            effectiveWeightSecondDeriv[j] = logSpaceDensitySecondDeriv[j];
        } else {
            effectiveWeight[j]            = floorValue;
            effectiveWeightDeriv[j]       = 0.0;
            effectiveWeightSecondDeriv[j] = 0.0;
        }
    }

    // Step 10b — renormalize in t-space after floor (floor adds mass)
    {
        double weightNormalization = 0.0;
        for (int j = 0; j < numLogGridPoints - 1; ++j)
            weightNormalization += 0.5 * (effectiveWeight[j] + effectiveWeight[j+1])
                                   * (logGridPoints[j+1] - logGridPoints[j]);
        if (weightNormalization > 0.0) {
            double normalizationInverse = 1.0 / weightNormalization;
            for (int j = 0; j < numLogGridPoints; ++j) {
                effectiveWeight[j]            *= normalizationInverse;
                effectiveWeightDeriv[j]       *= normalizationInverse;
                effectiveWeightSecondDeriv[j] *= normalizationInverse;
            }
        }
    }

    // Step 11 — store the quintic Hermite spline knots (evaluated at run time by
    // DenominatorDensity::evalW).
    DensityArrays out;
    out.logGridKnots          = std::move(logGridPoints);
    out.weightKnots           = std::move(effectiveWeight);
    out.weightDerivKnots      = std::move(effectiveWeightDeriv);
    out.weightSecondDerivKnots = std::move(effectiveWeightSecondDeriv);
    out.denominatorMin = denominatorMin;
    out.denominatorMax = denominatorMaxPhysical;
    out.ratio          = denominatorMaxPhysical / denominatorMin;
    out.logRatioMax    = physicalLogRatio;
    return out;
}

} // namespace detail
} // namespace minimax_cpppy
