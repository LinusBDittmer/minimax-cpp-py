#include <cassert>
#include <cmath>
#include "core/data_ext.hpp"

using minimax_cpppy::detail::data::ExtTableEntry;
using minimax_cpppy::detail::data::interpolateEntries;

// Synthetic non-uniform table: nlap=1, expon[0] = f(log10 r) = a known quadratic,
// weight[0] = a known linear. Nodes clustered low, sparse high.
static void test_nonuniform_exact_at_nodes_and_smooth() {
    const int nlap = 1, order = 5;
    const double lr[7] = {0.0, 0.05, 0.12, 0.3, 1.0, 3.0, 6.0};
    ExtTableEntry tbl[7]{};
    auto fE = [](double x){ return 2.0 + 0.5*x - 0.1*x*x; };
    auto fW = [](double x){ return -1.0 + 0.3*x; };
    for (int i = 0; i < 7; ++i) {
        tbl[i].range  = std::pow(10.0, lr[i]);
        tbl[i].errmax = 1e-3;
        tbl[i].expon[0]  = fE(lr[i]);
        tbl[i].weight[0] = fW(lr[i]);
    }
    // Exact recovery at a node.
    auto at = interpolateEntries(tbl, 7, nlap, tbl[3].range, order);
    assert(std::abs(at.expon[0]  - fE(lr[3])) < 1e-12);
    assert(std::abs(at.weight[0] - fW(lr[3])) < 1e-12);
    // Order-5 Lagrange reproduces a quadratic/linear exactly between nodes.
    const double xq = 0.5;
    auto mid = interpolateEntries(tbl, 7, nlap, std::pow(10.0, xq), order);
    assert(std::abs(mid.expon[0]  - fE(xq)) < 1e-9);
    assert(std::abs(mid.weight[0] - fW(xq)) < 1e-9);
    // Clamp below range: ratio 0.5 -> log10r approx -0.301, below lr[0]=0 -> clamps to node 0
    auto loc = interpolateEntries(tbl, 7, nlap, 0.5, order);
    assert(std::abs(loc.expon[0] - fE(0.0)) < 1e-9);

    // Clamp above range: ratio 1e10 -> log10r=10, above lr[6]=6.0 -> clamps to node 6
    auto hic = interpolateEntries(tbl, 7, nlap, 1e10, order);
    assert(std::abs(hic.expon[0] - fE(6.0)) < 1e-9);
}

int main() { test_nonuniform_exact_at_nodes_and_smooth(); return 0; }
