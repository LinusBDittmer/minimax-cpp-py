import numpy as np
import pytest
import minimax_cpppy as mx

NE_OCC  = np.array([-32.765635, -1.918798, -0.832097, -0.832097, -0.832097])
NE_VIRT = np.array([1.694558, 1.694558, 1.694558, 2.159425, 5.196711,
                    5.196711, 5.196711, 5.196711, 5.196711])

H2O_OCC  = np.array([-20.550438, -1.336658, -0.699262, -0.566562, -0.493142])
H2O_VIRT = np.array([0.185559, 0.256244, 0.789271, 0.854276, 1.163512,
                     1.200385, 1.253306, 1.444602, 1.476247, 1.674666,
                     1.867313, 1.934850, 2.452768, 2.490253, 3.285619,
                     3.338865, 3.510478, 3.865845, 4.147450])


# 7.4.1 — construction from numpy arrays
def test_denominator_density_construction():
    d = mx.DenominatorDensity(NE_OCC, NE_VIRT, bandwidth=0.1)
    assert d.ratio > 1.0
    assert d.delta_min > 0.0
    assert d.delta_max > d.delta_min


def _occ_virt():
    occ = np.array([-0.9, -0.5, -0.4])
    virt = np.array([0.2, 0.5, 1.0, 2.0])
    return occ, virt


def test_C_negative_raises():
    occ, virt = _occ_virt()
    with pytest.raises(ValueError):
        mx.DenominatorDensity(occ, virt, 1.0, C=-1.0)


def _ne_bounds():
    d = mx.DenominatorDensity(NE_OCC, NE_VIRT, bandwidth=1.0)
    return d.delta_min, d.delta_max


# 7.4.5 — biased_laplace smoke
def test_biased_laplace_smoke():
    ymin, ymax = _ne_bounds()
    expon, weight, errmax = mx.biased_laplace(
        5, ymin, ymax, NE_OCC, NE_VIRT, 1.0)
    assert len(expon) == 5
    assert len(weight) == 5
    assert all(e > 0 for e in expon)
    assert all(w > 0 for w in weight)
    assert errmax > 0
    assert np.isfinite(errmax)


# 7.4.6 — wrong ratio raises
def test_biased_wrong_ratio_raises():
    ymin, ymax = _ne_bounds()
    with pytest.raises((ValueError, RuntimeError)):
        mx.biased_laplace(5, ymin * 2.0, ymax, NE_OCC, NE_VIRT, 1.0)


# 7.4.6b — nlap/range validation (same contract as laplace_minimax)
def test_biased_nlap_zero_raises():
    ymin, ymax = _ne_bounds()
    with pytest.raises(ValueError):
        mx.biased_laplace(0, ymin, ymax, NE_OCC, NE_VIRT, 1.0)


def test_biased_nlap_too_large_raises():
    ymin, ymax = _ne_bounds()
    with pytest.raises(ValueError):
        mx.biased_laplace(31, ymin, ymax, NE_OCC, NE_VIRT, 1.0)


def test_biased_inverted_range_raises():
    ymin, ymax = _ne_bounds()
    with pytest.raises(ValueError):
        mx.biased_laplace(5, ymax, ymin, NE_OCC, NE_VIRT, 1.0)


def test_biased_nonpositive_ymin_raises():
    _, ymax = _ne_bounds()
    with pytest.raises(ValueError):
        mx.biased_laplace(5, -1.0, ymax, NE_OCC, NE_VIRT, 1.0)


# 7.4.7 — biased errmax stays within an order of magnitude of the unbiased
# result (correction, not a redesign) -- bias reduction itself is covered in
# tests/test_biased_minimax.cpp, which has access to the density internals.
def test_biased_errmax_comparable_to_unbiased():
    ymin, ymax = _ne_bounds()
    nlap = 4
    _, _, errmax_u = mx.laplace_minimax(nlap, ymin, ymax)
    _, _, errmax_b = mx.biased_laplace(
        nlap, ymin, ymax, NE_OCC, NE_VIRT, 1.0)
    assert np.isfinite(errmax_b)
    assert errmax_b < errmax_u * 10.0
