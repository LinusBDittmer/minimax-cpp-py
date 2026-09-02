import numpy as np
import pytest
from minimax_cpppy._minimax_core import DenominatorDensity

OCC  = np.array([-1.0, -0.5])
VIRT = np.array([ 1.0,  2.0])

def test_singles_delta_min():
    d = DenominatorDensity(OCC, VIRT, bandwidth=1.0, n_exc=1)
    # delta_min = 1 * (virt_min - occ_max) = 1 * (1.0 - (-0.5)) = 1.5
    assert abs(d.delta_min - 1.5) < 1e-10

def test_doubles_delta_min():
    d = DenominatorDensity(OCC, VIRT, bandwidth=1.0, n_exc=2)
    # delta_min = 2 * (1.0 - (-0.5)) = 3.0
    assert abs(d.delta_min - 3.0) < 1e-10

def test_triples_delta_min():
    d = DenominatorDensity(OCC, VIRT, bandwidth=1.0, n_exc=3)
    # delta_min = 3 * (virt_min - occ_max) = 3 * (1.0 - (-0.5)) = 4.5
    assert abs(d.delta_min - 4.5) < 1e-10

def test_triples_delta_max():
    d = DenominatorDensity(OCC, VIRT, bandwidth=1.0, n_exc=3)
    # delta_max = 3 * (virt_max - occ_min) = 3 * (2.0 - (-1.0)) = 9.0
    assert abs(d.delta_max - 9.0) < 1e-10

def test_invalid_n_exc_raises():
    with pytest.raises((ValueError, RuntimeError)):
        DenominatorDensity(OCC, VIRT, bandwidth=1.0, n_exc=0)

def test_default_is_doubles():
    d_default = DenominatorDensity(OCC, VIRT, bandwidth=1.0)
    d_two     = DenominatorDensity(OCC, VIRT, bandwidth=1.0, n_exc=2)
    assert abs(d_default.delta_min - d_two.delta_min) < 1e-14

def test_max_occ_kwarg_rejected():
    with pytest.raises(TypeError):
        DenominatorDensity(OCC, VIRT, bandwidth=1.0, max_occ=2)

def test_n_exc_4_constructs():
    d = DenominatorDensity(OCC, VIRT, bandwidth=1.0, n_exc=4)
    assert d.delta_min > 0.0
    assert d.delta_max >= d.delta_min
