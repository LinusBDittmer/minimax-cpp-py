// tools/dump_reference.cpp — dump current laplaceMinimax outputs as the regression oracle.
// Usage: dump_reference <out.csv>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "core/algorithm.hpp"

namespace minimax_cpppy {
namespace detail {
static double cuspLog10(int n) {
    return (4.4877 * std::sqrt((double)n) - 0.3845 * std::log((double)n) - 2.3654) / 2.302585092994046;
}
} // namespace detail
} // namespace minimax_cpppy

static std::vector<double> validationGrid(int nlap) {
    std::vector<double> g;
    for (int i = 0; i < 360; ++i) g.push_back(0.1 + (12.0 - 0.1) * i / 359.0);
    const double hi = minimax_cpppy::detail::cuspLog10(nlap) + 0.5;
    const double lo = 0.1;
    if (hi > lo) for (int i = 0; i < 40; ++i) g.push_back(lo + (hi - lo) * i / 39.0);
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return g;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "Usage: dump_reference <out.csv>\n"); return 1; }
    FILE* f = std::fopen(argv[1], "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::fprintf(f, "nlap,log10r,errmax,k,expon,weight\n");
    for (int nlap = 1; nlap <= 30; ++nlap) {
        for (double lr : validationGrid(nlap)) {
            const double ratio = std::pow(10.0, lr);
            auto r = minimax_cpppy::detail::laplaceMinimax(nlap, 1.0, ratio);
            for (int k = 0; k < nlap; ++k)
                std::fprintf(f, "%d,%.17g,%.17e,%d,%.17e,%.17e\n",
                             nlap, lr, r.errmax, k, r.expon[k], r.weight[k]);
        }
        std::fprintf(stderr, "nlap=%d done\n", nlap);
    }
    std::fclose(f);
    return 0;
}
