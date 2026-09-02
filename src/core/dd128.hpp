/**
 * @file dd128.hpp
 * @brief Double-double (128-bit) arithmetic, ported from Fortran dd128_arithmetics.F90.
 *
 * Ported from the Fortran reference by Benjamin Helmich-Paris.
 * Algorithms follow Shewchuk / Dekker / Knuth.
 *
 * Representation: DD{hi, lo} where the exact value is hi + lo,
 * with |lo| <= 0.5 ulp(hi).  All arithmetic operations maintain this invariant.
 */
#ifndef MINIMAX_CPPPY_DD128_HPP
#define MINIMAX_CPPPY_DD128_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace minimax_cpppy {
namespace detail {

/**
 * @brief Double-double floating-point number with ~31 decimal digits of precision.
 *
 * The exact value represented is hi + lo, where |lo| <= 0.5 ulp(hi).
 * All arithmetic operators preserve this invariant using error-free transformations.
 */
struct DD {
    double hi; ///< High-order part; dominant value.
    double lo; ///< Low-order correction; |lo| <= 0.5 ulp(hi).

    DD() noexcept : hi(0.0), lo(0.0) {}
    /** @brief Construct from a double-precision value with optional correction term. */
    explicit DD(double h, double l = 0.0) noexcept : hi(h), lo(l) {}

    // -----------------------------------------------------------------------
    // Helpers: twoSum, quickTwoSum, ddTwoProduct
    // -----------------------------------------------------------------------

    /** @brief Exact sum (a + b) = hi + lo using Shewchuk two-sum algorithm. */
    static DD twoSum(double a, double b) noexcept {
        double s = a + b;
        double v = s - a;
        double e = (a - (s - v)) + (b - v);
        return DD(s, e);
    }

    /** @brief Fast two-sum: requires |a| >= |b| (no branches). */
    static DD quickTwoSum(double a, double b) noexcept {
        // Requires |a| >= |b|
        double s = a + b;
        double e = b - (s - a);
        return DD(s, e);
    }

    /** @brief Exact product (a * b) = hi + lo via fused multiply-add (TwoProductFMA).
     *  std::fma is IEEE-correct on every target; it contracts to a hardware FMA
     *  where available and falls back to a correct (slower) libm call otherwise. */
    static DD ddTwoProduct(double a, double b) noexcept {
        double product = a * b;
        double error   = std::fma(a, b, -product);
        return DD(product, error);
    }

    // -----------------------------------------------------------------------
    // Arithmetic operators
    // -----------------------------------------------------------------------

    DD operator+(const DD& o) const noexcept {
        DD s  = twoSum(hi, o.hi);
        DD s2 = twoSum(lo, o.lo);
        s.lo += s2.hi;
        s = quickTwoSum(s.hi, s.lo);
        s.lo += s2.lo;
        s = quickTwoSum(s.hi, s.lo);
        return s;
    }

    DD& operator+=(const DD& o) noexcept { *this = *this + o; return *this; }

    DD operator-() const noexcept { return DD(-hi, -lo); }

    DD operator-(const DD& o) const noexcept { return *this + (-o); }

    DD& operator-=(const DD& o) noexcept { *this = *this - o; return *this; }

    DD operator*(const DD& o) const noexcept {
        DD p = ddTwoProduct(hi, o.hi);
        p.lo += hi * o.lo;
        p.lo += lo * o.hi;
        return quickTwoSum(p.hi, p.lo);
    }

    DD& operator*=(const DD& o) noexcept { *this = *this * o; return *this; }

    // DD * double — Fortran dd128_mul_doub
    DD operator*(double d) const noexcept {
        double prodHi    = hi * d;
        double prodHiErr = std::fma(hi, d, -prodHi);
        double prodLo    = lo * d;
        double sumHi     = prodHi + prodLo;
        double sumErr    = (prodLo - (sumHi - prodHi)) + prodHiErr;
        DD r = quickTwoSum(sumHi + sumErr, sumErr - ((sumHi + sumErr) - sumHi));
        return r;
    }

    friend DD operator*(double d, const DD& a) noexcept { return a * d; }
    DD& operator*=(double d) noexcept { *this = *this * d; return *this; }

    // DD / DD — Fortran dd128_div
    DD operator/(const DD& o) const noexcept {
        double reciprocal     = 1.0 / o.hi;
        double approxQuotient = hi * reciprocal;
        DD tmp1(approxQuotient, 0.0);
        DD tmp2 = o * tmp1;
        DD tmp3 = *this - tmp2;
        DD tmp4 = ddTwoProduct(reciprocal, tmp3.hi);
        tmp1.hi = approxQuotient; tmp1.lo = 0.0;
        return tmp1 + tmp4;
    }

    DD& operator/=(const DD& o) noexcept { *this = *this / o; return *this; }

    // DD / double — Fortran dd128_div_doub
    DD operator/(double d) const noexcept {
        double quotHi = hi / d;
        double prodHi        = quotHi * d;
        double prodHiErr     = std::fma(quotHi, d, -prodHi);
        double remainder     = hi - prodHi;
        double remError      = remainder - hi;
        double fullRemainder = ((-prodHi - remError) + (hi - (remainder - remError))) + lo - prodHiErr;
        double quotLo        = (remainder + fullRemainder) / d;
        return quickTwoSum(quotHi + quotLo, quotLo - ((quotHi + quotLo) - quotHi));
    }

    DD& operator/=(double d) noexcept { *this = *this / d; return *this; }

    // -----------------------------------------------------------------------
    // Comparison (lexicographic on {hi, lo})
    // -----------------------------------------------------------------------

    bool operator< (const DD& o) const noexcept {
        return hi < o.hi || (hi == o.hi && lo < o.lo);
    }
    bool operator<=(const DD& o) const noexcept {
        return hi < o.hi || (hi == o.hi && lo <= o.lo);
    }
    bool operator> (const DD& o) const noexcept { return o < *this; }
    bool operator>=(const DD& o) const noexcept { return o <= *this; }
    bool operator==(const DD& o) const noexcept { return hi == o.hi && lo == o.lo; }
    bool operator!=(const DD& o) const noexcept { return !(*this == o); }

    // -----------------------------------------------------------------------
    // Math functions
    // -----------------------------------------------------------------------

    /** @brief Absolute value: |a|. */
    static DD ddAbs(const DD& a) noexcept {
        return a.hi >= 0.0 ? a : -a;
    }

    /** @brief Double-double squaring: a² with full precision (Fortran dd128_sqr). */
    static DD ddSquare(const DD& a) noexcept {
        double productHi  = a.hi * a.hi;
        double productErr = std::fma(a.hi, a.hi, -productHi);
        productErr += 2.0 * (a.hi * a.lo);
        return quickTwoSum(productHi, productErr);
    }

    /** @brief Double-double square root: sqrt(a) (Fortran dd128_sqrt). */
    static DD ddSquareRoot(const DD& a) noexcept {
        double invSqrt    = 1.0 / std::sqrt(a.hi);
        double approxSqrt = a.hi * invSqrt;
        DD tmp1 = ddTwoProduct(approxSqrt, approxSqrt);
        DD tmp2 = a - tmp1;
        DD tmp3 = ddTwoProduct(invSqrt, tmp2.hi);
        tmp3.hi *= 0.5; tmp3.lo *= 0.5;
        DD result(approxSqrt, 0.0);
        return result + tmp3;
    }

    /**
     * @brief Double-double exponential: exp(a) via Taylor series with scaling-and-squaring (Fortran dd128_exp).
     * @param a Exponent; returns 0 for a < -512, and +inf for a > 700.
     */
    static DD ddExp(const DD& a) noexcept {
        static const DD XMIN(-512.0, 0.0);
        static const DD XMAX(700.0, 0.0);
        if (a < XMIN) { return DD(0.0, 0.0); }
        if (a > XMAX) { return DD(std::numeric_limits<double>::infinity(), 0.0); }

        // find scaleFactor = 2^scalingOrder such that a / scaleFactor < 1
        int64_t scaleFactor = 1;
        int     scalingOrder = 0;
        while (DD(static_cast<double>(scaleFactor), 0.0) <= a) {
            scaleFactor *= 2;
            ++scalingOrder;
        }
        // extra scaling for Taylor convergence
        scaleFactor  *= 256;
        scalingOrder += 8;

        // Convergence tolerance for the *reduced* series exp(a/scaleFactor),
        // which stays O(1) regardless of the original a -- NOT scaled by
        // exp(a.hi) (the original formula): squaring scalingOrder times
        // amplifies relative error by ~2^scalingOrder == scaleFactor, so the
        // per-term tolerance must shrink by that same factor to preserve
        // ~1e-30 final relative precision. The old formula grew UNBOUNDEDLY
        // with a.hi instead of shrinking, causing the loop to break after the
        // first term once exp(a.hi) exceeded ~1e30 (a.hi >~ 69).
        // scaleFactor is bounded to roughly [256, 2.6e5] given the XMIN/XMAX
        // guards above (a in (-512, 700]), so convergenceTol = 1e-30/scaleFactor
        // is always well within double's normal range -- no zero/underflow
        // guard needed here.
        double convergenceTol = 1e-30 / static_cast<double>(scaleFactor);

        // Taylor: exp(a/scaleFactor) = sum_{k=0}^inf (a/scaleFactor)^k / k!
        DD sum(0.0, 0.0);
        DD term(1.0, 0.0);
        for (int iter = 1; iter <= 1000; ++iter) {
            sum += term;
            if (std::abs(term.hi) < convergenceTol) { break; }
            DD nextTerm = term * a;
            nextTerm /= static_cast<double>(scaleFactor * static_cast<int64_t>(iter));
            term = nextTerm;
        }

        // squaring: sum^(2^scalingOrder)
        for (int i = 0; i < scalingOrder; ++i) {
            sum = ddSquare(sum);
        }

        return sum;
    }

    /**
     * @brief Double-double natural logarithm: log(a), refined to full DD precision.
     *
     * Seeds with the identity log(hi + lo) ≈ log(hi) + lo/hi, then applies one
     * Newton step on exp(y) = a (y -= 1 - a*exp(-y)) using the full-DD ddExp to
     * refine the seed from double precision (~1e-16) to full double-double
     * (~1e-30 relative), since std::log itself is only double-precision.
     * @param a Argument; must be positive and finite (checked in Debug builds).
     */
    static DD ddLog(const DD& a)
#ifndef MINIMAX_CPPPY_DEBUG_MODE__
    noexcept
#endif
    {
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
        if (a.hi <= 0.0 || !std::isfinite(a.hi)) {
            throw std::invalid_argument("ddLog: argument must be positive and finite");
        }
#endif
        double loghi = std::log(a.hi);
        double corr  = a.lo / a.hi;                 // corrects only the input lo part
        DD y = DD::twoSum(loghi, corr);
        // std::log above is only double-precision; one Newton step on exp(y)=a
        // refines y to full double-double (ddExp is full-DD):  y -= 1 - a*exp(-y).
        y = y - (DD(1.0) - a * DD::ddExp(-y));
        // ponytail: valid for |log a| < 512 (a < e^512); beyond that ddExp(-y) underflows
        // to 0 and this returns y-1. Far outside any Laplace-quadrature input; no guard.
        return y;
    }
};

// -----------------------------------------------------------------------
// DD128 linear algebra helpers (flat arrays, stride-1 assumed)
// -----------------------------------------------------------------------

/** @brief Double-double Euclidean norm: result = ||x||_2 over n elements. */
inline DD ddNorm2(int n, const DD* x) noexcept {
    DD s(0.0, 0.0);
    for (int i = 0; i < n; ++i) { s += DD::ddSquare(x[i]); }
    return DD::ddSquareRoot(s);
}

/** @brief In-place scale: x := alpha * x over n elements. */
inline void ddScale(int n, const DD& alpha, DD* x) noexcept {
    for (int i = 0; i < n; ++i) { x[i] *= alpha; }
}

/** @brief Copy n elements: dst := src. */
inline void ddCopy(int n, const DD* src, DD* dst) noexcept {
    for (int i = 0; i < n; ++i) { dst[i] = src[i]; }
}

/**
 * @brief Compute y := alpha * A^T * x + beta * y (DD matrix-vector product, transposed).
 *
 * A is stored column-major with leading dimension lda.
 * @param m Number of rows of A.
 * @param n Number of columns of A (length of output y).
 * @param alpha Scalar multiplier.
 * @param A     Column-major matrix (m × n) with leading dim lda.
 * @param lda   Leading dimension of A.
 * @param x     Input vector of length m.
 * @param beta  Scalar multiplier for y.
 * @param y     In/out vector of length n.
 */
inline void ddGemvTransposed(int m, int n, const DD& alpha, const DD* A, int lda,
                              const DD* x, const DD& beta, DD* y) noexcept {
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("ddGemvTransposed: non-positive dimension");
    }
    if (!A || !x || !y) {
        throw std::invalid_argument("ddGemvTransposed: null pointer argument");
    }
#endif
    // y_j = alpha * sum_i A[i,j] * x[i]  (j=0..n-1, i=0..m-1)
    // A[i,j] = A[i + j*lda] (column-major)
    for (int j = 0; j < n; ++j) {
        DD dotProduct(0.0, 0.0);
        for (int i = 0; i < m; ++i) {
            dotProduct += A[i + j * lda] * x[i];
        }
        y[j] = alpha * dotProduct + beta * y[j];
    }
}

/**
 * @brief LU factorisation with partial pivoting in-place (Fortran dd128_ludcmp, Crout).
 *
 * On exit: A is overwritten with its LU factorisation; pivotIndices records row pivots.
 *
 * Not noexcept: in MINIMAX_CPPPY_DEBUG_MODE__ builds, a near-zero pivot throws
 * std::runtime_error rather than silently producing a singular factorisation --
 * a caller whose retry logic depends on catching that (e.g. newtonDampedSolve's
 * Levenberg-damping escalation) needs it to propagate normally, not terminate
 * the process (see newton.hpp).
 *
 * @param A            Column-major n × n matrix; overwritten with LU on exit.
 * @param n            Matrix order.
 * @param pivotIndices Output pivot index array of length n.
 */
inline void ddLuFactorize(DD* A, int n, int* pivotIndices) {
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (n <= 0) {
        throw std::invalid_argument("ddLuFactorize: non-positive n");
    }
    if (!A || !pivotIndices) {
        throw std::invalid_argument("ddLuFactorize: null pointer argument");
    }
#endif
    // A is column-major n×n
    for (int j = 0; j < n; ++j) {
        // find pivot row
        int pivotRow  = j;
        DD  maxPivot  = DD::ddAbs(A[j + j*n]);
        for (int i = j+1; i < n; ++i) {
            DD candidate = DD::ddAbs(A[i + j*n]);
            if (candidate > maxPivot) { maxPivot = candidate; pivotRow = i; }
        }
        // swap rows pivotRow <-> j
        if (pivotRow != j) {
            for (int k = 0; k < n; ++k) {
                DD tmp          = A[j       + k*n];
                A[j       + k*n] = A[pivotRow + k*n];
                A[pivotRow + k*n] = tmp;
            }
        }
        pivotIndices[j] = pivotRow;
        // Crout elimination
        DD pivot = A[j + j*n];
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
        if (DD::ddAbs(pivot).hi < 1e-280) {
            throw std::runtime_error(
                "ddLuFactorize: near-zero pivot at column " + std::to_string(j) +
                " (|pivot|=" + std::to_string(DD::ddAbs(pivot).hi) +
                "); Jacobian may be singular");
        }
#endif
        for (int i = j+1; i < n; ++i) {
            A[i + j*n] /= pivot;
            DD multiplier = A[i + j*n];
            for (int k = j+1; k < n; ++k) {
                A[i + k*n] -= multiplier * A[j + k*n];
            }
        }
    }
}

/**
 * @brief LU backsubstitution to solve A*x = b (Fortran dd128_lubksb).
 *
 * Must be called after ddLuFactorize.  rhs is overwritten with the solution.
 * @param A            LU-factorised column-major matrix from ddLuFactorize (not modified).
 * @param n            Matrix order.
 * @param pivotIndices Pivot indices from ddLuFactorize.
 * @param rhs          Right-hand-side on entry; solution on exit.
 */
inline void ddLuSolve(const DD* A, int n, const int* pivotIndices, DD* rhs) noexcept {
#ifdef MINIMAX_CPPPY_DEBUG_MODE__
    if (n <= 0) {
        throw std::invalid_argument("ddLuSolve: non-positive n");
    }
    if (!A || !pivotIndices || !rhs) {
        throw std::invalid_argument("ddLuSolve: null pointer argument");
    }
#endif
    // forward substitution with pivoting
    for (int i = 0; i < n; ++i) {
        int pivotIdx  = pivotIndices[i];
        DD  temp      = rhs[pivotIdx];
        rhs[pivotIdx] = rhs[i];
        rhs[i]        = temp;
        for (int j = 0; j < i; ++j) {
            rhs[i] -= A[i + j*n] * rhs[j];
        }
    }
    // backward substitution
    for (int i = n-1; i >= 0; --i) {
        for (int j = i+1; j < n; ++j) {
            rhs[i] -= A[i + j*n] * rhs[j];
        }
        rhs[i] /= A[i + i*n];
    }
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_DD128_HPP
