// Regression test: reduced ext tables vs the golden reference fixture.
//
// Mirrors tools/check_against_reference.cpp's split gate (design decision
// 2026-06-21), as an assert-based CTest reading the fixture from the source tree:
//   * Resolved region   (full-table quadrature error >= OVERRESOLVED_THRESH):
//     points are Remez-polished truth -> point-wise relative deviation of
//     expon/weight vs the reference must be <= RESOLVED_TOL.
//   * Over-resolved region (full-table quadrature error < OVERRESOLVED_THRESH):
//     points come straight from table interpolation (no polish) and are
//     near-degenerate, so point-wise comparison only measures interpolation
//     noise. Instead require the ACTUAL quadrature error of the returned points
//     stays essentially exact (<= ESS_EXACT).
//
// Region is classified by computeInitialError of the full-table reference
// points, NOT the stored errmax column (an unreliable interpolated field).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>
#include "core/algorithm.hpp"

#ifndef MINIMAX_REF_CSV
#define MINIMAX_REF_CSV "tests/data/reference_outputs.csv"
#endif

static constexpr double OVERRESOLVED_THRESH = 1e-9;
static constexpr double ESS_EXACT           = 1e-8;
static constexpr double RESOLVED_TOL        = 1e-7;

int main(int argc, char** argv) {
    // Optional args: nlap_min nlap_max lr_min lr_max (log10 ratio bounds, half-open [lr_min, lr_max)).
    const int    nlap_min = argc > 1 ? std::atoi(argv[1])  : 1;
    const int    nlap_max = argc > 2 ? std::atoi(argv[2])  : 30;
    const double lr_min   = argc > 3 ? std::atof(argv[3])  : -1e9;
    const double lr_max   = argc > 4 ? std::atof(argv[4])  :  1e9;

    using minimax_cpppy::detail::DD;
    FILE* f = std::fopen(MINIMAX_REF_CSV, "r");
    assert(f && "reference_outputs.csv not found");
    char line[256];
    std::fgets(line, sizeof line, f); // header

    struct Key { int nlap; double lr; bool operator<(const Key& o) const {
        return nlap != o.nlap ? nlap < o.nlap : lr < o.lr; } };
    std::map<Key, std::vector<std::pair<double,double>>> ref;
    while (std::fgets(line, sizeof line, f)) {
        int nlap, k; double lr, errmax, e, w;
        if (std::sscanf(line, "%d,%lf,%lf,%d,%lf,%lf", &nlap, &lr, &errmax, &k, &e, &w) == 6)
            if (nlap >= nlap_min && nlap <= nlap_max && lr >= lr_min && lr < lr_max)
                ref[{nlap, lr}].emplace_back(e, w);
    }
    std::fclose(f);

    double worst_p = 0.0, worst_q = 0.0;
    bool saw_smallR_highn = false;
    for (auto& kv : ref) {
        const int nlap = kv.first.nlap;
        const double lr = kv.first.lr;
        const double ratio = std::pow(10.0, lr);
        const auto& ew = kv.second;
        assert((int)ew.size() == nlap && "fixture key has wrong row count");
        const int nsamp = 40 * nlap + 1;

        // Classify by actual quadrature error of the full-table reference points.
        std::vector<DD> ef(nlap), wf(nlap);
        for (int k = 0; k < nlap; ++k) { ef[k] = DD(ew[k].first); wf[k] = DD(ew[k].second); }
        const double q_full =
            minimax_cpppy::detail::computeInitialError(ef.data(), wf.data(), nlap, ratio, nsamp).hi;

        auto r = minimax_cpppy::detail::laplaceMinimax(nlap, 1.0, ratio);

        if (q_full < OVERRESOLVED_THRESH) {
            std::vector<DD> e(nlap), w(nlap);
            for (int k = 0; k < nlap; ++k) { e[k] = DD(r.expon[k]); w[k] = DD(r.weight[k]); }
            double q = minimax_cpppy::detail::computeInitialError(e.data(), w.data(), nlap, ratio, nsamp).hi;
            worst_q = std::max(worst_q, q);
            assert(q <= ESS_EXACT && "over-resolved quadrature error exceeds 1e-8");
        } else {
            for (int k = 0; k < nlap; ++k) {
                double de = std::abs(r.expon[k]  - ew[k].first ) / std::max(std::abs(ew[k].first ), 1e-300);
                double dw = std::abs(r.weight[k] - ew[k].second) / std::max(std::abs(ew[k].second), 1e-300);
                double d = std::max(de, dw);
                worst_p = std::max(worst_p, d);
                assert(d <= RESOLVED_TOL && "resolved table exceeds 1e-7 vs reference");
            }
        }
        if (nlap >= 20 && lr <= 2.0) saw_smallR_highn = true; // ensure coverage exists
    }
    // Only assert coverage when the tested range overlaps nlap>=20 and lr<=2.
    if (nlap_max >= 20 && lr_min < 2.0)
        assert(saw_smallR_highn && "fixture lacks small-R nlap>=20 coverage");
    std::printf("table regression OK, resolved worst_rel=%.3e, over-resolved worst_q=%.3e\n",
                worst_p, worst_q);
    return 0;
}
