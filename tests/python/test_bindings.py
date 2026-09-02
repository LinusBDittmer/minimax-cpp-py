import numpy as np
import pytest

import minimax_cpppy as mm


# ── Smoke ──────────────────────────────────────────────────────────────────

def test_smoke():
    expon, weight, errmax = mm.laplace_minimax(5, 0.2, 40.0)
    assert expon is not None
    assert weight is not None
    assert isinstance(errmax, float)


# ── Output shape and dtype ─────────────────────────────────────────────────

def test_output_shape_nlap5():
    expon, weight, _ = mm.laplace_minimax(5, 0.2, 40.0)
    assert expon.shape == (5,)
    assert weight.shape == (5,)


def test_output_shape_nlap10():
    expon, weight, _ = mm.laplace_minimax(10, 0.2, 40.0)
    assert expon.shape == (10,)
    assert weight.shape == (10,)


def test_output_dtype():
    expon, weight, _ = mm.laplace_minimax(5, 0.2, 40.0)
    assert expon.dtype == np.float64
    assert weight.dtype == np.float64


# ── Numerics ───────────────────────────────────────────────────────────────

def _max_approx_error(expon, weight, ymin, ymax, ngrid=1000):
    ts = np.linspace(0.0, 1.0, ngrid)
    xs = ymin * (ymax / ymin) ** ts
    approx = np.sum(
        weight[np.newaxis, :] * np.exp(-expon[np.newaxis, :] * xs[:, np.newaxis]),
        axis=1,
    )
    return float(np.max(np.abs(1.0 / xs - approx)))


def test_nlap3_approximation_quality():
    expon, weight, _ = mm.laplace_minimax(3, 1.0, 100.0)
    err = _max_approx_error(expon, weight, 1.0, 100.0)
    assert err < 1e-2, f"nlap=3 error {err:.3e} >= 1e-2"


def test_nlap5_approximation_quality():
    expon, weight, _ = mm.laplace_minimax(5, 1.0, 100.0)
    err = _max_approx_error(expon, weight, 1.0, 100.0)
    assert err < 1e-3, f"nlap=5 error {err:.3e} >= 1e-3"


def test_nlap7_approximation_quality():
    expon, weight, _ = mm.laplace_minimax(7, 1.0, 100.0)
    err = _max_approx_error(expon, weight, 1.0, 100.0)
    assert err < 1e-4, f"nlap=7 error {err:.3e} >= 1e-4"


def test_errmax_field_is_positive():
    _, _, errmax = mm.laplace_minimax(5, 1.0, 100.0)
    assert errmax > 0.0


def test_error_decreases_with_nlap():
    prev = float("inf")
    for nlap in [3, 5, 7, 10, 15]:
        _, _, errmax = mm.laplace_minimax(nlap, 1.0, 100.0)
        assert errmax < prev, f"errmax did not decrease at nlap={nlap}"
        prev = errmax


# ── Error handling ─────────────────────────────────────────────────────────

def test_nlap_too_large_raises():
    with pytest.raises((ValueError, RuntimeError)):
        mm.laplace_minimax(54, 1.0, 100.0)


def test_nlap_zero_raises():
    with pytest.raises((ValueError, RuntimeError)):
        mm.laplace_minimax(0, 1.0, 100.0)


def test_inverted_range_raises():
    with pytest.raises((ValueError, RuntimeError)):
        mm.laplace_minimax(5, 100.0, 1.0)


def test_equal_range_raises():
    with pytest.raises((ValueError, RuntimeError)):
        mm.laplace_minimax(5, 1.0, 1.0)


def test_nonpositive_ymin_raises():
    with pytest.raises(ValueError):
        mm.laplace_minimax(5, -1.0, 100.0)


def test_zero_ymin_raises():
    with pytest.raises(ValueError):
        mm.laplace_minimax(5, 0.0, 100.0)


# ── Initial guess ──────────────────────────────────────────────────────────

def test_init_guess_accepted():
    expon0, weight0, _ = mm.laplace_minimax(5, 1.0, 100.0)
    expon1, weight1, errmax1 = mm.laplace_minimax(
        5, 1.0, 100.0, init_expon=expon0, init_weight=weight0)
    assert expon1.shape == (5,)
    assert weight1.shape == (5,)
    assert errmax1 > 0.0


def test_init_guess_result_quality():
    expon0, weight0, _ = mm.laplace_minimax(5, 1.0, 100.0)
    expon1, weight1, _ = mm.laplace_minimax(
        5, 1.0, 100.0, init_expon=expon0, init_weight=weight0)
    err = _max_approx_error(expon1, weight1, 1.0, 100.0)
    assert err < 1e-3, f"init_guess result error {err:.3e} >= 1e-3"


def test_init_guess_only_expon_raises():
    expon0, _, _ = mm.laplace_minimax(5, 1.0, 100.0)
    with pytest.raises((ValueError, TypeError)):
        mm.laplace_minimax(5, 1.0, 100.0, init_expon=expon0)


def test_init_guess_only_weight_raises():
    _, weight0, _ = mm.laplace_minimax(5, 1.0, 100.0)
    with pytest.raises((ValueError, TypeError)):
        mm.laplace_minimax(5, 1.0, 100.0, init_weight=weight0)


def test_init_guess_wrong_size_raises():
    bad = np.array([1.0, 2.0])  # size 2, nlap=5
    with pytest.raises((ValueError, RuntimeError)):
        mm.laplace_minimax(5, 1.0, 100.0, init_expon=bad, init_weight=bad)


def test_init_guess_distant_seed_converges():
    # Seed from R=100 run, apply to R=1e4 interval — tests hint path for far-off seed
    expon_seed, weight_seed, _ = mm.laplace_minimax(5, 1.0, 100.0)
    expon1, weight1, errmax1 = mm.laplace_minimax(
        5, 1.0, 1e4, init_expon=expon_seed, init_weight=weight_seed)
    assert expon1.shape == (5,)
    assert errmax1 > 0.0
    err = _max_approx_error(expon1, weight1, 1.0, 1e4)
    assert err < 1.0, f"distant-seed result error {err:.3e} >= 1.0"


def test_init_guess_positional_args_still_work():
    # nlap, ymin, ymax are positional; init_* are keyword-only
    expon0, weight0, _ = mm.laplace_minimax(5, 1.0, 100.0)
    expon1, weight1, _ = mm.laplace_minimax(5, 1.0, 100.0,
                                             init_expon=expon0,
                                             init_weight=weight0)
    assert expon1.shape == (5,)
