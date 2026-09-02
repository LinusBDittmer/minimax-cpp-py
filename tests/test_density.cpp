#include "test_helpers.hpp"
#include "minimax_cpppy/biasing.hpp"
#include "molecular_orbital_data.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

// ── helper ────────────────────────────────────────────────────────────────────
static double integrate_pt(const minimax_cpppy::DenominatorDensity& d,
                            int nsamp = 4000)
{
    // Trapezoidal integral of evalW over [0, ln(ratio)]
    double tmax = std::log(d.ratio());
    double dt   = tmax / (nsamp - 1);
    double sum  = 0.0;
    double w0, dw0, d2w0;
    d.evalW(0.0, w0, dw0, d2w0);
    double wprev = w0;
    for (int j = 1; j < nsamp; ++j) {
        double t = j * dt;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        sum   += 0.5 * (wprev + w) * dt;
        wprev  = w;
    }
    return sum;
}

// 7.1.1 — single occ/virt pair: Δ = 2*(ε_v - ε_o) exactly
MINIMAX_TEST(density_single_pair_delta_min) {
    double occ[]  = { -1.0 };
    double virt[] = {  2.0 };
    minimax_cpppy::DenominatorDensity d(occ, 1, virt, 1, 1.0, 4096, 256, 1e-3);
    double expected_delta_min = 2.0 * (virt[0] - occ[0]); // = 6.0
    MINIMAX_REQUIRE(std::abs(d.deltaMin() - expected_delta_min) < 1e-10);
    MINIMAX_REQUIRE(d.deltaMax() > d.deltaMin());
    MINIMAX_REQUIRE(d.ratio() > 1.0);
    MINIMAX_REQUIRE(std::abs(d.ratio() - d.deltaMax() / d.deltaMin()) < 1e-10);
}

// 7.1.3 — normalization: ∫ p_t dt ≈ 1 for various (h, n_fft, n_t)
MINIMAX_TEST(density_normalization_ne) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3);
    double integral = integrate_pt(d, 5000);
    MINIMAX_REQUIRE(std::abs(integral - 1.0) < 2e-3);
}

MINIMAX_TEST(density_normalization_h2o) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3);
    double integral = integrate_pt(d, 5000);
    MINIMAX_REQUIRE(std::abs(integral - 1.0) < 2e-3);
}

// 7.1.8 — phase alignment: shift all energies by δ → p_t invariant
MINIMAX_TEST(density_phase_alignment) {
    double occ[]  = { -1.0, -0.5 };
    double virt[] = {  1.0,  2.0 };
    const double delta = 3.7;
    double occ2[]  = { occ[0]  + delta, occ[1]  + delta };
    double virt2[] = { virt[0] + delta, virt[1] + delta };

    minimax_cpppy::DenominatorDensity d1(occ,  2, virt,  2, 1.0, 2048, 256, 1e-3);
    minimax_cpppy::DenominatorDensity d2(occ2, 2, virt2, 2, 1.0, 2048, 256, 1e-3);

    double tmax = std::log(d1.ratio());
    for (int j = 0; j <= 50; ++j) {
        double t = tmax * j / 50.0;
        double w1, dw1, d2w1, w2, dw2, d2w2;
        d1.evalW(t, w1, dw1, d2w1);
        d2.evalW(t, w2, dw2, d2w2);
        MINIMAX_REQUIRE(std::abs(w1 - w2) < 1e-8);
    }
}

// 7.1.9 — large system: no NaN/Inf
MINIMAX_TEST(density_large_system_no_nan) {
    const auto& m = mol_data::MOLECULES[4];  // benzene (21 occ, 93 virt)
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3);
    double tmax = std::log(d.ratio());
    for (int j = 0; j <= 100; ++j) {
        double t = tmax * j / 100.0;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        MINIMAX_REQUIRE(std::isfinite(w));
        MINIMAX_REQUIRE(std::isfinite(dw));
        MINIMAX_REQUIRE(std::isfinite(d2w));
        MINIMAX_REQUIRE(w >= 0.0);
    }
}

// 7.1.2 — symmetry check
MINIMAX_TEST(density_symmetry) {
    double occ[]  = { -3.0, -2.0, -1.0 };
    double virt[] = {  1.0,  2.0,  3.0 };
    minimax_cpppy::DenominatorDensity d(occ, 3, virt, 3, 1.0, 4096, 512, 0.0);
    double integral = integrate_pt(d, 5000);
    MINIMAX_REQUIRE(std::abs(integral - 1.0) < 5e-3);
}

// 7.1.4 — floor application
MINIMAX_TEST(density_floor_application) {
    double occ[]  = { -1.0 };
    double virt[] = {  2.0 };
    double floor_frac = 0.05;
    minimax_cpppy::DenominatorDensity d(occ, 1, virt, 1, 1.0, 4096, 256, floor_frac);

    double tmax = std::log(d.ratio());
    double w_max = 0.0;
    for (int j = 0; j <= 200; ++j) {
        double t = tmax * j / 200.0;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        if (w > w_max) w_max = w;
    }
    for (int j = 0; j <= 200; ++j) {
        double t = tmax * j / 200.0;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        MINIMAX_REQUIRE(w >= floor_frac * w_max - 1e-12);
    }
}

// 7.1.5 — evalW finite-difference derivative check
MINIMAX_TEST(density_derivative_fd_check) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3);
    double tmax = std::log(d.ratio());
    double delta = 1e-5;
    for (int j = 1; j <= 49; ++j) {
        double t = tmax * j / 50.0;
        double wm, dwm, d2wm, wp, dwp, d2wp, w0, dw0, d2w0;
        d.evalW(t - delta, wm, dwm, d2wm);
        d.evalW(t + delta, wp, dwp, d2wp);
        d.evalW(t,         w0, dw0, d2w0);

        double fd_dw  = (wp - wm) / (2.0 * delta);
        double fd_d2w = (wp - 2.0*w0 + wm) / (delta * delta);

        double ref_dw = std::max(std::abs(dw0), 1e-6);
        MINIMAX_REQUIRE(std::abs(dw0 - fd_dw)   / ref_dw < 1e-5);
        MINIMAX_REQUIRE(std::abs(d2w0 - fd_d2w) < 1e-4 + 1e-4 * std::abs(d2w0));
    }
}

// 7.1.6 — C² continuity at spline knots
MINIMAX_TEST(density_c2_continuity) {
    const auto& m = mol_data::MOLECULES[0];  // Ne
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 2048, 64, 1e-3);
    double tmax = std::log(d.ratio());
    const int ncheck = 200;
    double w_prev, dw_prev, d2w_prev;
    d.evalW(0.0, w_prev, dw_prev, d2w_prev);
    for (int j = 1; j <= ncheck; ++j) {
        double t = tmax * j / ncheck;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        MINIMAX_REQUIRE(std::abs(w - w_prev) < 10.0 * (w_prev + w + 1e-15));
        w_prev = w; dw_prev = dw; d2w_prev = d2w;
    }
}

// 7.1.7 — evalW clamp behaviour
MINIMAX_TEST(density_evalW_clamp) {
    double occ[]  = { -1.0 };
    double virt[] = {  2.0 };
    minimax_cpppy::DenominatorDensity d(occ, 1, virt, 1, 1.0, 2048, 128, 1e-3);
    double w_lo, dw_lo, d2w_lo;
    double w_hi, dw_hi, d2w_hi;
    double w0, dw0, d2w0;
    double tmax = std::log(d.ratio());
    d.evalW(-1.0,        w_lo, dw_lo, d2w_lo);
    d.evalW(tmax + 10.0, w_hi, dw_hi, d2w_hi);
    d.evalW(0.0,         w0,  dw0,  d2w0);
    d.evalW(tmax,        w_hi, dw_hi, d2w_hi);
    MINIMAX_REQUIRE(std::isfinite(w_lo) && w_lo >= 0.0);
    MINIMAX_REQUIRE(std::isfinite(w_hi) && w_hi >= 0.0);
    MINIMAX_REQUIRE(std::abs(w_lo - w0) < 1e-14);
}

// 7.1.10 — bandwidth sensitivity
// Use H2O (multi-orbital) so delta_min < delta_max_phys and the FFT pipeline runs.
// Single-pair systems trigger the degenerate path regardless of bandwidth.
MINIMAX_TEST(density_bandwidth_sensitivity) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    // Use bandwidths far enough apart that the narrow one (sigma = 0.025*delta_max)
    // always produces a sharper peak than the wide one (sigma = 0.5*delta_max).
    minimax_cpppy::DenominatorDensity d_narrow(m.occ, m.n_occ, m.virt, m.n_virt,
                                               0.1, 4096, 512, 0.0);
    minimax_cpppy::DenominatorDensity d_wide  (m.occ, m.n_occ, m.virt, m.n_virt,
                                               2.0, 4096, 512, 0.0);

    double tmax_n = std::log(d_narrow.ratio());
    double tmax_w = std::log(d_wide.ratio());

    double wmax_n = 0.0, wmax_w = 0.0;
    for (int j = 0; j <= 300; ++j) {
        double wn, dwn, d2wn, ww, dww, d2ww;
        d_narrow.evalW(tmax_n * j / 300.0, wn, dwn, d2wn);
        d_wide  .evalW(tmax_w * j / 300.0, ww, dww, d2ww);
        if (wn > wmax_n) wmax_n = wn;
        if (ww > wmax_w) wmax_w = ww;
    }
    MINIMAX_REQUIRE(wmax_n > wmax_w);
}

// n_exc: bounds scaling for singles, doubles, triples
MINIMAX_TEST(density_n_exc_bounds_scaling) {
    double occ[]  = { -1.0 };
    double virt[] = {  2.0 };
    // singles: delta_min = 1*(2 - (-1)) = 3
    minimax_cpppy::DenominatorDensity d1(occ, 1, virt, 1, 1.0, 4096, 256, 1e-3,
                                         -1.0, /*C=*/0.0, /*n_exc=*/1);
    MINIMAX_REQUIRE(std::abs(d1.deltaMin() - 3.0) < 1e-10);

    // doubles: delta_min = 2*(2 - (-1)) = 6
    minimax_cpppy::DenominatorDensity d2(occ, 1, virt, 1, 1.0, 4096, 256, 1e-3,
                                         -1.0, /*C=*/0.0, /*n_exc=*/2);
    MINIMAX_REQUIRE(std::abs(d2.deltaMin() - 6.0) < 1e-10);

    // triples: delta_min = 3*(2 - (-1)) = 9
    minimax_cpppy::DenominatorDensity d3(occ, 1, virt, 1, 1.0, 4096, 256, 1e-3,
                                         -1.0, /*C=*/0.0, /*n_exc=*/3);
    MINIMAX_REQUIRE(std::abs(d3.deltaMin() - 9.0) < 1e-10);
}

// n_exc: normalization for singles (n_exc=1)
MINIMAX_TEST(density_n_exc_normalization_singles) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3,
                                        -1.0, /*C=*/0.0, /*n_exc=*/1);
    double integral = integrate_pt(d, 5000);
    MINIMAX_REQUIRE(std::abs(integral - 1.0) < 2e-3);
}

// n_exc: normalization for triples (n_exc=3)
MINIMAX_TEST(density_n_exc_normalization_triples) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3,
                                        -1.0, /*C=*/0.0, /*n_exc=*/3);
    double integral = integrate_pt(d, 5000);
    MINIMAX_REQUIRE(std::abs(integral - 1.0) < 2e-3);
}

// n_exc=4: finite and non-negative for benzene
MINIMAX_TEST(density_n_exc_4_finite_nonneg) {
    const auto& m = mol_data::MOLECULES[4];  // benzene
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3,
                                        -1.0, /*C=*/0.0, /*n_exc=*/4);
    double tmax = std::log(d.ratio());
    for (int j = 0; j <= 100; ++j) {
        double t = tmax * j / 100.0;
        double w, dw, d2w;
        d.evalW(t, w, dw, d2w);
        MINIMAX_REQUIRE(std::isfinite(w));
        MINIMAX_REQUIRE(w >= 0.0);
    }
}

// n_exc=0: invalid argument throws
MINIMAX_TEST(density_n_exc_invalid_throws) {
    double occ[]  = { -1.0 };
    double virt[] = {  2.0 };
    MINIMAX_REQUIRE_THROW_TYPE(
        minimax_cpppy::DenominatorDensity(occ, 1, virt, 1, 1.0, 4096, 256, 1e-3,
                                          -1.0, /*C=*/0.0, /*n_exc=*/0),
        std::invalid_argument);
}

// n_exc=3 bounds: delta_min = 3 * D_single_min, delta_max = 3 * D_single_max
// D_single_min = virt_min - occ_max = 1.0 - (-0.5) = 1.5 → delta_min = 3 * 1.5 = 4.5
// D_single_max = virt_max - occ_min = 2.0 - (-1.0) = 3.0 → delta_max = 3 * 3.0 = 9.0
MINIMAX_TEST(density_n_exc_triples_rhf_bounds) {
    double occ[]  = { -1.0, -0.5 };
    double virt[] = {  1.0,  2.0 };
    minimax_cpppy::DenominatorDensity d(occ, 2, virt, 2, 1.0, 4096, 256, 1e-3,
                                        -1.0, /*C=*/0.0, /*n_exc=*/3);
    MINIMAX_REQUIRE(std::abs(d.deltaMin() - 4.5) < 1e-10);
    MINIMAX_REQUIRE(std::abs(d.deltaMax() - 9.0) < 1e-10);
}

// Jacobian: w(t) = p_Δ(Δ)·Δ grows with t for large bandwidth (nearly flat p_Δ)
MINIMAX_TEST(density_jacobian_weight_increases_with_t) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    // Very large bandwidth → nearly uniform p_Δ → w(t) ≈ C·Δ_min·e^t, strictly increasing.
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        10.0, 4096, 512, 1e-3);
    double tmax = std::log(d.ratio());
    double w0, dw0, d2w0, wend, dwend, d2wend;
    d.evalW(0.0,  w0,   dw0,   d2w0);
    d.evalW(tmax, wend, dwend, d2wend);
    MINIMAX_REQUIRE(wend > w0);
}

// 7.1.x — constant density offset C
MINIMAX_TEST(density_offset_zero_is_noop) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    // Same construction; only C differs (C defaults to 0 via the floor_frac-only form).
    minimax_cpppy::DenominatorDensity d0(m.occ, m.n_occ, m.virt, m.n_virt,
                                         1.0, 4096, 512, 1e-3);
    minimax_cpppy::DenominatorDensity dC(m.occ, m.n_occ, m.virt, m.n_virt,
                                         1.0, 4096, 512, 1e-3, -1.0,
                                         /*C=*/0.0);
    const double tmax = std::log(d0.ratio());
    for (int i = 1; i < 20; ++i) {
        double t = tmax * i / 20.0;
        double w0, dw0, d2w0, wC, dwC, d2wC;
        d0.evalW(t, w0, dw0, d2w0);
        dC.evalW(t, wC, dwC, d2wC);
        MINIMAX_REQUIRE_CLOSE(w0, wC, 1e-15);
    }
}

MINIMAX_TEST(density_offset_large_C_flattens) {
    const auto& m = mol_data::MOLECULES[1];  // H2O
    // Large C: pedestal dominates, so w(t) is near-constant across the interval.
    minimax_cpppy::DenominatorDensity d(m.occ, m.n_occ, m.virt, m.n_virt,
                                        1.0, 4096, 512, 1e-3, -1.0,
                                        /*C=*/1e6);
    const double tmax = std::log(d.ratio());
    double w1, w2, dw, d2w;
    d.evalW(0.25 * tmax, w1, dw, d2w);
    d.evalW(0.75 * tmax, w2, dw, d2w);
    MINIMAX_REQUIRE_CLOSE(w1 / w2, 1.0, 1e-3);
}

int main() { MINIMAX_RUN_TESTS(); }
