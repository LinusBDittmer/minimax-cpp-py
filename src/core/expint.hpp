#ifndef MINIMAX_CPPPY_EXPINT_HPP
#define MINIMAX_CPPPY_EXPINT_HPP

#include "dd128.hpp"

namespace minimax_cpppy {
namespace detail {

// Euler-Mascheroni constant to double-double precision (hi + lo == gamma to ~1e-33).
// lo = gamma - hi exactly. Verified by expint1_series_and_cf_agree_to_dd / known-value tests.
inline const DD& ddEulerGamma() {
    static const DD g(0.57721566490153287, -4.94291515243064510e-18);
    return g;
}

// E1(z) via Taylor series: E1(z) = -gamma - ln z - sum_{n>=1} (-1)^n z^n/(n*n!).
// Accurate for small-to-moderate z (z < 1 in dispatch); loses digits to
// cancellation for large z. Exposed for testing against the CF branch.
inline DD ddExpInt1Series(const DD& z) {
    DD sum = -ddEulerGamma() - DD::ddLog(z);
    DD term(1.0, 0.0);        // z^n / n! accumulated below (starts as z^0/0!)
    DD acc(0.0, 0.0);         // sum_{n>=1} (-1)^n z^n/(n*n!)
    for (int n = 1; n <= 200; ++n) {
        term = term * z / DD(static_cast<double>(n));   // z^n / n!
        DD contrib = term / DD(static_cast<double>(n)); // z^n /(n*n!)
        if (n & 1) acc = acc - contrib; else acc = acc + contrib;
        if (std::fabs(contrib.hi) <= 1e-34 * std::fabs(acc.hi)) break;
    }
    return sum - acc;
}

// E1(z) via modified-Lentz continued fraction:
//   E1(z) = e^{-z} / (z + 1 - 1^2/(z + 3 - 2^2/(z + 5 - ...))).
// Converges for z >= 1 (fast for z >~ 2; ~327 iters at z≈1). Exposed for testing.
inline DD ddExpInt1CF(const DD& z) {
    const DD tiny(1e-300, 0.0);
    DD b = z + DD(1.0);
    DD c = DD(1.0) / tiny;    // "very large"
    DD d = DD(1.0) / b;
    DD h = d;
    for (int i = 1; i <= 500; ++i) {   // z near the z=1 dispatch boundary needs ~327 iters
        const DD ai(static_cast<double>(-i * i));
        b = b + DD(2.0);
        d = b + ai * d; if (d.hi == 0.0) d = tiny;
        d = DD(1.0) / d;
        c = b + ai / c; if (c.hi == 0.0) c = tiny;
        DD del = c * d;
        h = h * del;
        if (std::fabs((del - DD(1.0)).hi) <= 1e-31) break;   // DD-aware: keeps sub-eps part
    }
    DD emz = DD::ddExp(-z);
    return emz * h;   // underflows to DD(0) when e^{-z} underflows (large z)
}

// E1(z) for z > 0: series below 1, continued fraction at/above 1.
inline DD ddExpInt1(const DD& z) {
    return (z.hi < 1.0) ? ddExpInt1Series(z) : ddExpInt1CF(z);
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_EXPINT_HPP
