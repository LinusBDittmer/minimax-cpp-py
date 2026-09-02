# tests/python/test_minimax_lp.py
import numpy as np
import pytest
from minimax_cpppy import laplace_lp


def test_smoke():
    expon, weight, ln_norm = laplace_lp(4, 1.0, 100.0, 4)
    assert expon.shape == (4,)
    assert weight.shape == (4,)
    assert np.all(np.isfinite(expon)) and np.all(expon > 0)
    assert np.all(np.isfinite(weight))
    assert np.isfinite(ln_norm) and ln_norm > 0


def test_odd_n_runs():
    expon, weight, ln_norm = laplace_lp(4, 1.0, 100.0, 3)
    assert expon.shape == (4,)
    assert np.all(np.isfinite(expon)) and np.all(expon > 0)
    assert np.all(np.isfinite(weight))
    assert np.isfinite(ln_norm) and ln_norm > 0


def test_n1_runs():
    expon, weight, ln_norm = laplace_lp(4, 1.0, 100.0, 1)
    assert expon.shape == (4,)
    assert np.all(np.isfinite(expon)) and np.all(expon > 0)
    assert np.isfinite(ln_norm) and ln_norm > 0


@pytest.mark.parametrize("n", [1.5, 2.5, 3.5])
def test_noninteger_n_runs(n):
    expon, weight, ln_norm = laplace_lp(4, 1.0, 100.0, n)
    assert expon.shape == (4,)
    assert np.all(np.isfinite(expon)) and np.all(expon > 0)
    assert np.all(np.isfinite(weight))
    assert np.isfinite(ln_norm) and ln_norm > 0


def test_noninteger_n_between_neighbors():
    # A non-integer order's optimum sits monotonically between its integer
    # neighbors in the reported L_n norm (integral norm over [0, ln R]).
    _, _, norm2 = laplace_lp(4, 1.0, 100.0, 2.0)
    _, _, norm25 = laplace_lp(4, 1.0, 100.0, 2.5)
    _, _, norm3 = laplace_lp(4, 1.0, 100.0, 3.0)
    assert norm2 > norm25 > norm3


def test_n_too_small_raises():
    with pytest.raises(ValueError):
        laplace_lp(4, 1.0, 100.0, 0)
