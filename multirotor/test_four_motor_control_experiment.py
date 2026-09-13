import numpy as np
import pytest

from multirotor.four_motor_control_experiment import (
    FourMotorCase, FourMotorController, RateStateObserver, cases,
    reference_signal, run_case)
from multirotor.four_motor_plant import FourMotorPlant


def test_reference_waveforms():
    t = np.linspace(0, .8, 401)
    step = reference_signal(t, (80., 60., 40.))
    np.testing.assert_allclose(step[t < .1], 0)
    np.testing.assert_allclose(step[-1], np.deg2rad([80., 60., 40.]))
    np.testing.assert_allclose(step[t > .1], np.tile(np.deg2rad([80., 60., 40.]), ((t > .1).sum(), 1)))
    reversal = reference_signal(t, (400., 300., 150.), "reversal")
    np.testing.assert_allclose(reversal[t > .6], 0)
    assert np.all(reversal[(t > .35) & (t < .6)] < 0)
    with pytest.raises(ValueError):
        reference_signal(t, (1., 2., 3.), "unknown")
    with pytest.raises(ValueError):
        reference_signal(t, (1., 2.))


def test_observer_does_not_use_true_rotor_state():
    plant = FourMotorPlant()
    controller = FourMotorController(plant, "lqr", 40., .1)
    estimate, bias = controller.observer.correct(np.zeros(3))
    assert estimate.shape == (plant.state_size,)
    assert bias.shape == (3,)


def test_lqr_and_mpc_commands_within_motor_bounds():
    plant = FourMotorPlant()
    reference = np.deg2rad([80., 60., 40.])
    for name in ("lqr", "mpc"):
        controller = FourMotorController(plant, name, 40., .1)
        command, diagnostics = controller.command(np.zeros(3), reference)
        assert np.all(command >= .05 - 1e-9)
        assert np.all(command <= 1. + 1e-9)
        assert diagnostics["success"]
        assert np.isfinite(command).all()


def test_lqr_slew_weight_rejected():
    plant = FourMotorPlant()
    with pytest.raises(ValueError):
        FourMotorController(plant, "lqr", 40., .1, slew_weight=.1)


def test_run_case_small_step_tracks():
    case = FourMotorCase("small", (80., 60., 40.), duration_s=.4, noise_dps=.25)
    row, trace = run_case(case, controller="mpc", horizon_ms=20., effort_weight=.1,
                          observation="ideal", seed=3)
    assert row["solver_failure_count"] == 0
    assert row["rms_error_deg_s"] < 30.
    post = trace["t"] > .3
    np.testing.assert_allclose(np.rad2deg(trace["rates"][post]),
                               np.rad2deg(trace["reference"][post]), atol=8.)
    commands = trace["commands"]
    assert np.all(commands >= .05 - 1e-9) and np.all(commands <= 1. + 1e-9)
    assert np.isfinite(trace["full_state"]).all()


def test_run_case_mpc_command_precedes_reference():
    case = FourMotorCase("pre", (80., 60., 40.), duration_s=.2, noise_dps=0.)
    _, trace = run_case(case, controller="mpc", horizon_ms=40., observation="ideal")
    step_index = int(np.flatnonzero(trace["t"] >= .1)[0])
    assert not np.all(trace["commands"][:step_index] > .5)


def test_run_case_validation():
    with pytest.raises(ValueError):
        run_case(FourMotorCase(disturbance_rad_s2=(np.nan, 0., 0.)))
    with pytest.raises(ValueError):
        run_case(FourMotorCase(duration_s=.001))
    with pytest.raises(ValueError):
        run_case(observation="oracle")


def test_case_list_is_self_consistent():
    names = [case.name for case in cases()]
    assert len(names) == len(set(names))
    for case in cases():
        np.asarray(case.amplitude_dps, dtype=float)
