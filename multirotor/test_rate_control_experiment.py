import numpy as np
import pytest

from multirotor.rate_control_experiment import (
    RateCase, RateObserver, rate_state_space, reference_signal, run_case, tracking_target,
)


def test_reference_is_causal_and_reversal_returns_to_zero():
    t = np.array([0., .099, .1, .349, .35, .599, .6])
    np.testing.assert_allclose(np.rad2deg(reference_signal(t, 100., "reversal")),
                               [0., 0., 100., 100., -100., -100., 0.])
    with pytest.raises(ValueError):
        reference_signal(t, 100., "unknown")


def test_tracking_target_is_disturbed_equilibrium():
    A, B, _, D = rate_state_space(1200., .016, .002)
    target, command = tracking_target(2., 30., 1200., len(A))
    np.testing.assert_allclose(A @ target + B[:, 0] * command + D * 30., target,
                               rtol=1e-12, atol=1e-12)


def test_rate_observer_estimates_constant_torque_bias():
    A, B, C, D = rate_state_space(1200., .016, .002)
    observer = RateObserver(A, B, C, D, .002, .25)
    x = np.zeros(len(A))
    for _ in range(1000):
        estimated, bias = observer.correct(float((C @ x)[0]))
        observer.predict(0.)
        x = A @ x + D * 20.
    assert bias == pytest.approx(20., abs=.5)
    np.testing.assert_allclose(estimated[:2], (x - D * 20.)[:2], atol=.05)


@pytest.mark.parametrize("controller", ["pid", "lqr", "mpc"])
def test_rollout_is_finite_bounded_and_tracks(controller):
    row, trace = run_case(RateCase(duration_s=.35, noise_dps=0.), controller=controller)
    assert np.isfinite(trace["rates"]).all()
    assert np.isfinite(trace["commands"]).all()
    assert np.max(np.abs(trace["commands"])) <= row["command_scale"] + 1e-7
    assert trace["rates"][-1] == pytest.approx(np.deg2rad(105.), abs=.15)
    assert row["reference_preview"] is False


def test_ideal_small_signal_mpc_matches_lqr_across_horizons():
    case = RateCase(amplitude_dps=1., duration_s=.25, noise_dps=0.)
    _, baseline = run_case(case, controller="lqr", observation="ideal")
    for horizon in [10., 40., 80.]:
        _, trace = run_case(case, controller="mpc", horizon_ms=horizon, observation="ideal")
        np.testing.assert_allclose(trace["rates"], baseline["rates"], atol=2e-4)
        assert trace["solver_success"].all()


def test_reference_does_not_generate_pre_step_action():
    _, trace = run_case(RateCase(duration_s=.2, noise_dps=0.), observation="ideal")
    np.testing.assert_allclose(trace["commands"][trace["t"] < .1], 0., atol=1e-10)
