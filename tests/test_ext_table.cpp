#include <cassert>
#include <cmath>
#include <vector>
#include "core/algorithm.hpp"
#include "core/data_ext.hpp"

// Test: maehlySolver with exact extrema as warm-start produces same result
static void test_maehly_warmstart() {
    const int nlap = 5;
    const double ratio = 100.0;

    auto ref = minimax_cpppy::detail::laplaceMinimax(nlap, 1.0, ratio);

    std::vector<minimax_cpppy::detail::DD> exp_dd(nlap), w_dd(nlap);
    for (int k = 0; k < nlap; ++k) {
        exp_dd[k] = minimax_cpppy::detail::DD(ref.expon[k]);
        w_dd[k]   = minimax_cpppy::detail::DD(ref.weight[k]);
    }

    const int numExtrema = 2 * nlap - 1;
    std::vector<minimax_cpppy::detail::DD> ext_cold(numExtrema), ext_warm(numExtrema);
    minimax_cpppy::detail::maehlySolver(
        ext_cold.data(), numExtrema,
        minimax_cpppy::detail::DD(1.0), minimax_cpppy::detail::DD(ratio),
        200, 1e-10, exp_dd.data(), w_dd.data(), nlap, 0.3, 1e-6,
        nullptr);

    std::vector<double> guess_hi(numExtrema);
    for (int i = 0; i < numExtrema; ++i) { guess_hi[i] = ext_cold[i].hi; }
    minimax_cpppy::detail::maehlySolver(
        ext_warm.data(), numExtrema,
        minimax_cpppy::detail::DD(1.0), minimax_cpppy::detail::DD(ratio),
        200, 1e-10, exp_dd.data(), w_dd.data(), nlap, 0.3, 1e-6,
        guess_hi.data());

    for (int i = 0; i < numExtrema; ++i) {
        assert(std::abs(ext_cold[i].hi - ext_warm[i].hi) < 1e-12);
    }
}

// Guards remezLoop's finalExtremaOut capture, which tools/gen_table relies on
// to store the warm-start extrema in the ext tables.
static void test_remezloop_captures_extrema() {
    const int nlap = 3;
    const double ratio = 50.0;

    auto seed = minimax_cpppy::detail::data::interpolatedLookup(nlap, ratio);

    std::vector<minimax_cpppy::detail::DD> exp_dd(nlap), w_dd(nlap);
    for (int k = 0; k < nlap; ++k) {
        exp_dd[k] = minimax_cpppy::detail::DD(seed.expon[k]);
        w_dd[k]   = minimax_cpppy::detail::DD(seed.weight[k]);
    }
    minimax_cpppy::detail::DD erramp(std::abs(seed.errmax));

    std::vector<double> extrema;
    auto result = minimax_cpppy::detail::remezLoop(
        nlap, 1.0, ratio,
        std::move(exp_dd), std::move(w_dd), erramp,
        200, 1e-10, 1e-15, 0.3, 1e-6, 1e-4,
        nullptr, &extrema);

    assert(static_cast<int>(extrema.size()) == 2 * nlap - 1);
    for (double e : extrema) {
        (void)e;
        assert(e > 1.0 && e < ratio);
    }
    for (int i = 1; i < (int)extrema.size(); ++i) {
        assert(extrema[i] > extrema[i-1]);
    }
    (void)result;
}

static bool extTablePopulated() {
    const auto* t = minimax_cpppy::detail::data::extTableForNlap(1);
    return t != nullptr && t[0].errmax != 0.0;
}

static void test_ext_table_skeleton() {
    (void)minimax_cpppy::detail::data::extTableForNlap(1);
}

static void test_interpolation_at_grid_point() {
    if (!extTablePopulated()) return;
    const int nlap = 1;
    const auto* tbl = minimax_cpppy::detail::data::extTableForNlap(nlap);
    const int mid = minimax_cpppy::detail::data::extTableSizeForNlap(nlap) / 2;

    // Lagrange interpolation evaluated exactly at a stored node returns that node.
    auto interp = minimax_cpppy::detail::data::interpolatedLookup(nlap, tbl[mid].range);

    assert(std::abs(interp.expon[0] - tbl[mid].expon[0]) < 1e-12 * std::abs(tbl[mid].expon[0]));
    assert(std::abs(interp.weight[0] - tbl[mid].weight[0]) < 1e-12 * std::abs(tbl[mid].weight[0]));
    assert(std::abs(interp.extrema[0] - tbl[mid].extrema[0]) < 1e-12 * std::abs(tbl[mid].extrema[0]));
}

static void test_extended_range_convergence() {
    if (!extTablePopulated()) return;
    struct Case { int nlap; double ymin, ymax; };
    Case cases[] = {
        {  5,  1.0,  1e8  },
        { 10,  1.0,  1e6  },
        { 20,  0.1,  1e9  },
        { 27,  1.0,  1e10 },
    };
    for (auto& c : cases) {
        auto result = minimax_cpppy::laplaceMinimax(c.nlap, c.ymin, c.ymax);
        for (int k = 0; k < c.nlap; ++k) {
            assert(result.expon[k]  > 0.0);
            assert(result.weight[k] > 0.0);
        }
        assert(result.errmax > 0.0 && result.errmax < 1.0);
        if (c.nlap >= 20) {
            assert(result.errmax < 1e-8);
        }
    }
}

static void test_old_table_ratios_still_converge() {
    struct Case { int nlap; double ymin, ymax; double max_errmax; };
    Case cases[] = {
        { 1,  1.0,  1.1,  2e-3  },  // nlap=1/R=1.1 knife-edge: true minimax errmax ~1.07e-3
        { 5,  1.0,  1e3,  1e-3  },  // table seed 6.4e-4
        {20,  1.0,  1e6,  1e-7  },  // table seed 2.9e-8
        {30,  1.0,  1e12, 2e-12 },  // table seed for nlap=30 is ~1e-12; use 2x margin
    };
    for (auto& c : cases) {
        auto result = minimax_cpppy::laplaceMinimax(c.nlap, c.ymin, c.ymax);
        assert(result.errmax < c.max_errmax);
        for (int k = 0; k < c.nlap; ++k) {
            assert(result.expon[k] > 0.0);
            assert(result.weight[k] > 0.0);
        }
    }
}

static void test_nlap_28_30_ext_tables() {
    // Verify that ext tables for nlap=28..30 are present, non-zero,
    // and that laplaceMinimax converges for representative inputs.
    for (int nlap : {28, 29, 30}) {
        {
            const auto* tbl = minimax_cpppy::detail::data::extTableForNlap(nlap);
            const int mid = minimax_cpppy::detail::data::extTableSizeForNlap(nlap) / 2;
            assert(tbl != nullptr);
            assert(tbl[0].errmax != 0.0);
            assert(tbl[mid].errmax != 0.0);
        }

        // Spot-check: interpolatedLookup returns plausible data.
        {
            auto entry = minimax_cpppy::detail::data::interpolatedLookup(nlap, 100.0);
            for (int k = 0; k < nlap; ++k) {
                assert(entry.expon[k]  > 0.0);
                assert(entry.weight[k] > 0.0);
            }
        }

        // Full convergence check at a moderate ratio.
        auto result = minimax_cpppy::laplaceMinimax(nlap, 1.0, 100.0);
        assert(result.errmax > 0.0 && result.errmax < 1.0);
        for (int k = 0; k < nlap; ++k) {
            assert(result.expon[k]  > 0.0);
            assert(result.weight[k] > 0.0);
        }
    }
}

int main() {
    test_maehly_warmstart();
    test_remezloop_captures_extrema();
    test_ext_table_skeleton();
    test_interpolation_at_grid_point();
    test_extended_range_convergence();
    test_old_table_ratios_still_converge();
    test_nlap_28_30_ext_tables();
    return 0;
}
