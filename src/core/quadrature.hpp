#ifndef MINIMAX_CPPPY_QUADRATURE_HPP
#define MINIMAX_CPPPY_QUADRATURE_HPP

#include <vector>

namespace minimax_cpppy {
namespace detail {

struct QuadRule {
    std::vector<double> t;  // absolute nodes
    std::vector<double> w;  // absolute weights
};

// 16-point Gauss–Legendre rule on [-1,1], stored as 8 symmetric (node,weight) pairs.
inline const double* gl16Nodes() {
    static const double x[8] = {
        0.0950125098376374, 0.2816035507792589, 0.4580167776572274,
        0.6178762444026438, 0.7554044083550030, 0.8656312023878318,
        0.9445750230732326, 0.9894009349916499};
    return x;
}
inline const double* gl16Weights() {
    static const double w[8] = {
        0.1894506104550685, 0.1826034150449236, 0.1691565193950025,
        0.1495959888165767, 0.1246289712555339, 0.0951585116824928,
        0.0622535239386479, 0.0271524594117541};
    return w;
}

// ponytail: fixed order 16; convergence is driven by panel doubling, not order.
// Add Golub–Welsch for arbitrary order only if a future caller needs it.
inline QuadRule compositeGaussLegendre(double a, double b, int panels) {
    if (panels < 1) panels = 1;
    const double* gx = gl16Nodes();
    const double* gw = gl16Weights();
    QuadRule r;
    r.t.reserve(static_cast<size_t>(16 * panels));
    r.w.reserve(static_cast<size_t>(16 * panels));
    const double H = (b - a) / panels;
    for (int p = 0; p < panels; ++p) {
        const double c = a + p * H;
        const double mid = c + 0.5 * H;
        const double half = 0.5 * H;
        for (int j = 0; j < 8; ++j) {
            const double off = half * gx[j];
            const double ww  = half * gw[j];
            r.t.push_back(mid - off); r.w.push_back(ww);
            r.t.push_back(mid + off); r.w.push_back(ww);
        }
    }
    return r;
}

} // namespace detail
} // namespace minimax_cpppy

#endif // MINIMAX_CPPPY_QUADRATURE_HPP
