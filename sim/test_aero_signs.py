"""Sign checks on the strip aerodynamics (sim/wing_model.py, which
test_wing_equiv.py ties to the C++ TailsitterPhysics).

Aero frame: x forward, y right, z down; +roll moment = right wing down,
+pitch = nose up, +yaw = nose right.
"""

import numpy as np
import pytest

from sim import wing_model as wm


@pytest.fixture(scope="module")
def model():
    return wm.load_as_built()


def moments(model, u, v, w):
    p, strips = model
    return wm.forces_moments(p, strips, np.array([u, v, w], float), np.zeros(3),
                             0.0, 0.0, (0.0, 0.0))[1]


def test_sideslip_rolls_away_and_weathervanes(model):
    """Wind from the right (v > 0): the swept-back wing rolls left (dihedral
    effect from sweep) and yaws its nose toward the wind."""
    V, alpha, beta = 12.0, np.radians(2.0), np.radians(10.0)
    u = V * np.cos(beta) * np.cos(alpha)
    v = V * np.sin(beta)
    w = V * np.cos(beta) * np.sin(alpha)
    m_slip = moments(model, u, v, w)
    m_none = moments(model, V * np.cos(alpha), 0.0, V * np.sin(alpha))
    assert abs(m_none[0]) < 1e-9 and abs(m_none[2]) < 1e-9, "symmetric flight is symmetric"
    assert m_slip[0] < 0, f"roll should be left, got {m_slip}"
    assert m_slip[2] > 0, f"yaw should be toward the wind, got {m_slip}"


def test_backward_flight_is_pitch_unstable(model):
    """Flying backwards, a nose-up perturbation is reinforced: in reversed
    flow the force acts at 3/4 chord, upstream of the CG."""
    V = 12.0
    # flying backwards (u < 0) with the nose 5 deg up relative to the flow:
    # the relative wind then comes from slightly below the trailing edge
    a = np.radians(5.0)
    m_up = moments(model, -V * np.cos(a), 0.0, -V * np.sin(a))
    m_dn = moments(model, -V * np.cos(a), 0.0, V * np.sin(a))
    # unstable: the moment pushes further in the direction of the perturbation
    assert np.sign(m_up[1]) != np.sign(m_dn[1])
    stiffness = (m_up[1] - m_dn[1]) / (2 * a)
    forward = (moments(model, V * np.cos(a), 0.0, V * np.sin(a))[1]
               - moments(model, V * np.cos(a), 0.0, -V * np.sin(a))[1]) / (2 * a)
    assert forward < 0, "forward flight should be pitch-stable"
    assert stiffness * forward < 0, "backward flight should have the opposite stiffness"
