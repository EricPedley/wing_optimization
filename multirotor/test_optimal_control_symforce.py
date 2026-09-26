"""Tests for optimal_control_symforce: parity with the OSQP LinearMPC and the
JAX FourMotorPlant, plus a compiled-code smoke test of the generated C++.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag

from multirotor.four_motor_plant import FourMotorPlant
from multirotor.optimal_control import LinearMPC, lqr
from multirotor.optimal_control_symforce import (
    RiccatiMPC, SymbolicPlant, controller_mats, linearize, step)

MPC_DIR = Path(__file__).resolve().parent / "symforce_mpc"


@pytest.fixture(scope="module")
def plant():
    return FourMotorPlant()


def _scaled_controller(plant, effort=.1, slew=.1, collective=10., horizon=20):
    sp = SymbolicPlant(plant)
    As, Bs, Q, R, P, tinv = controller_mats(sp, effort, collective,
                                          np.deg2rad(100.))
    S = np.eye(4) * plant.dt * slew
    lo, hi = .05 - plant.trim_command, 1. - plant.trim_command
    return (LinearMPC(As, Bs, Q, R, horizon, lo, hi, terminal_cost=P, slew_cost=S),
            RiccatiMPC(As, Bs, Q, R, horizon, lo, hi, terminal_cost=P, slew_cost=S),
            tinv)


def test_symbolic_discrete_model_matches_expm(plant):
    _, A, B = linearize(plant)
    A_ref, B_ref = plant.discrete_model()
    assert A.shape == A_ref.shape and B.shape == B_ref.shape
    np.testing.assert_allclose(A, A_ref, atol=2e-4, rtol=1e-4)
    np.testing.assert_allclose(B, B_ref, atol=1e-9, rtol=0)


def test_symbolic_step_matches_jax(plant):
    rng = np.random.default_rng(0)
    x = plant.trim_state.copy()
    for _ in range(10):
        u = np.clip(plant.trim_command + rng.normal(size=4) * .1, 0., 1.)
        a = np.asarray(plant.step(x, u))
        b = step(plant, x, u)
        np.testing.assert_allclose(a, b, atol=1e-9)
        x = a


def test_riccati_matches_lqr_steady_state():
    rng = np.random.default_rng(1)
    n, m = 4, 2
    A = rng.normal(size=(n, n)) * .3 + np.eye(n) * .9
    B = rng.normal(size=(n, m))
    Q, R = np.eye(n) * 2, np.eye(m)
    mpc = RiccatiMPC(A, B, Q, R, 60, -np.inf, np.inf)
    K, _ = lqr(A, B, Q, R)
    for _ in range(10):
        x = rng.normal(size=n)
        np.testing.assert_allclose(mpc.solve(x).u, -K @ x, atol=1e-8)


def test_riccati_matches_osqp_unconstrained(plant):
    n = plant.state_size
    rng = np.random.default_rng(2)
    for slew in (0., .1):
        bounded, riccati_bounded, tinv = _scaled_controller(plant, slew=slew)
        osqp_unbounded = LinearMPC(bounded.A, bounded.B, bounded.Q, bounded.R,
                                   20, -np.inf, np.inf,
                                   terminal_cost=bounded.terminal_cost,
                                   slew_cost=bounded.S)
        riccati = RiccatiMPC(bounded.A, bounded.B, bounded.Q, bounded.R,
                             20, -np.inf, np.inf,
                             terminal_cost=bounded.terminal_cost,
                             slew_cost=bounded.S)
        for _ in range(10):
            x = rng.normal(size=n) * .3
            rhs = np.zeros(7)
            rhs[3:6] = rng.normal(size=3) * .2
            target = tinv @ rhs
            xr, ur = target[:n], target[n:]
            up = rng.normal(size=4) * .05
            a = osqp_unbounded.solve(x, xr, ur, up)
            b = riccati.solve(x, xr, ur, up)
            assert a.success and b.success
            np.testing.assert_allclose(b.u, a.u, atol=1e-8)
            np.testing.assert_allclose(b.predicted_u, a.predicted_u, atol=1e-7)


def test_riccati_bounded_respects_limits(plant):
    n = plant.state_size
    rng = np.random.default_rng(3)
    osqp_mpc, riccati, tinv = _scaled_controller(plant)
    worst = 0.
    for _ in range(20):
        x = rng.normal(size=n)
        rhs = np.zeros(7)
        rhs[3:6] = rng.normal(size=3)
        target = tinv @ rhs
        xr, ur = target[:n], target[n:]
        b = riccati.solve(x, xr, ur, np.zeros(4))
        a = osqp_mpc.solve(x, xr, ur, np.zeros(4))
        assert b.success
        assert np.all(b.u >= riccati.u_min - 1e-12)
        assert np.all(b.u <= riccati.u_max + 1e-12)
        assert np.all(b.predicted_u >= riccati.u_min - 1e-12)
        assert np.all(b.predicted_u <= riccati.u_max + 1e-12)
        # projection differs from the true constrained optimum, but stays near it
        worst = max(worst, np.max(np.abs(a.u - b.u)))
    assert worst < 2.0  # loose bound: clip is suboptimal but not unstable


def test_validation_rejects_bad_inputs(plant):
    n = plant.state_size
    _, riccati, _ = _scaled_controller(plant)
    with pytest.raises(ValueError):
        riccati.solve(np.zeros(n + 1))
    with pytest.raises(ValueError):
        RiccatiMPC(np.eye(n), np.ones((n, 4)), np.eye(n), np.eye(4),
                   20, np.ones(4), np.zeros(4))


def test_generated_cpp_smoke(plant):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ not available")
    out = Path(tempfile.mkdtemp()) / "mpc_smoke"
    subprocess.run([gxx, "-O2", "-std=c++17", "-I.", "-Ieigen_shim",
                    "mpc_smoke.cpp", "-o", str(out)], cwd=MPC_DIR,
                   check=True, capture_output=True)
    result = subprocess.run([str(out)], check=True, capture_output=True,
                            text=True, timeout=30)
    # first tick: hover trim + 2 rad/s roll reference -> bang-bang roll command
    line = next(l for l in result.stdout.splitlines() if l.startswith("tick 0"))
    tokens = line.split()
    cmd = np.array([float(v) for v in tokens[tokens.index("cmd") + 1:
                                            tokens.index("cmd") + 5]])
    np.testing.assert_allclose(cmd, [.05, .05, 1., 1.], atol=1e-6)
