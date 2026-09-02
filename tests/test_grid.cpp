#include <cassert>
#include <cmath>
#include "core/grid.hpp"

using minimax_cpppy::detail::cuspLog10;
using minimax_cpppy::detail::GridParams;
using minimax_cpppy::detail::buildLog10Grid;

static void test_grid_invariants() {
    GridParams p{60, 40, 1.5, 1.0};      // n_dense, n_sparse, half_width, margin
    for (int nlap : {4, 12, 24, 30}) {
        auto g = buildLog10Grid(nlap, p, 0.1, 12.0);
        // sorted strictly increasing
        for (size_t i = 1; i < g.size(); ++i) assert(g[i] > g[i-1]);
        // endpoints present, in bounds
        assert(std::abs(g.front() - 0.1) < 1e-12);
        assert(std::abs(g.back() - 12.0) < 1e-12);
        // nodes cluster AT the cusp: the smallest adjacent gap sits near the
        // cusp and is much finer than the overall mean spacing. (Comparing
        // below-vs-above-cusp halves is wrong: for high nlap the cusp sits near
        // the high end, so the below-cusp half spans a large sparsely-covered
        // range and its mean spacing exceeds the above-cusp half's.)
        const double c = cuspLog10(nlap);
        if (c > 0.5 && c < 11.5) {
            double min_gap = 1e9, min_mid = 0.0;
            for (size_t i = 1; i < g.size(); ++i) {
                double gap = g[i] - g[i-1];
                if (gap < min_gap) { min_gap = gap; min_mid = 0.5*(g[i]+g[i-1]); }
            }
            const double mean_gap = (g.back() - g.front()) / (g.size() - 1);
            assert(std::abs(min_mid - c) < p.half_width_dec); // tightest spacing near cusp
            assert(min_gap < 0.25 * mean_gap);                // genuinely clustered
        }
    }
}

int main() { test_grid_invariants(); return 0; }
