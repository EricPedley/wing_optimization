from types import SimpleNamespace

import numpy as np
import pytest

from multirotor.optimal_control import LinearMPC, augment_input_delay, discretize, lqr


def scalar_mpc(**overrides):
    settings = dict(A=[[1.]], B=[[1.]], Q=[[1.]], R=[[1.]], horizon=3,
                    u_min=-1., u_max=1.)
    return LinearMPC(**(settings | overrides))


def test_exact_zoh_integrators():
    A, B = discretize([[0., 1.], [0., 0.]], [[0.], [2.]], .2)
    np.testing.assert_allclose(A, [[1., .2], [0., 1.]], atol=1e-14)
    np.testing.assert_allclose(B, [[.04], [.4]], atol=1e-14)
    A, B = discretize(np.zeros((2, 2)), np.eye(2), .07)
    np.testing.assert_allclose(A, np.eye(2))
    np.testing.assert_allclose(B, .07 * np.eye(2))


def test_exact_zoh_first_order_lag():
    dt, tau, gain = .013, .08, 2.7
    A, B = discretize([[-1 / tau]], [[gain / tau]], dt)
    np.testing.assert_allclose(A, [[np.exp(-dt / tau)]], atol=1e-14)
    np.testing.assert_allclose(B, [[gain * (1 - np.exp(-dt / tau))]], atol=1e-14)


@pytest.mark.parametrize("delay", [0, 1, 2, 4])
def test_delay_causality_and_oldest_first_queue(delay):
    A, B = np.eye(2), np.diag([1., 2.])
    Ad, Bd = augment_input_delay(A, B, delay)
    state = np.zeros(2 + 2 * delay)
    commands = [np.array([k + 1., -k - 1.]) for k in range(8)]
    physical = np.zeros(2)
    queue = [np.zeros(2) for _ in range(delay)]
    for command in commands:
        applied = queue.pop(0) if delay else command
        physical += B @ applied
        if delay:
            queue.append(command)
        state = Ad @ state + Bd @ command
        np.testing.assert_allclose(state[:2], physical)
        if delay:
            np.testing.assert_allclose(state[2:], np.concatenate(queue))


def test_lqr_stabilizes_discrete_system():
    A, B = discretize([[0., 1.], [1., .1]], [[0.], [1.]], .1)
    Q, R = np.diag([4., .2]), np.array([[.3]])
    K, P = lqr(A, B, Q, R)
    assert np.max(np.abs(np.linalg.eigvals(A - B @ K))) < 1
    np.testing.assert_allclose(P, Q + A.T @ P @ A - A.T @ P @ B @ K, atol=1e-10)
    np.testing.assert_allclose(P, P.T, atol=1e-14)


@pytest.mark.parametrize("horizon", [1, 2, 5, 12])
def test_unconstrained_mpc_matches_lqr(horizon):
    A, B = discretize([[0., 1.], [0., 0.]], [[0.], [1.]], .1)
    Q, R = np.diag([3., .4]), np.array([[.2]])
    K, P = lqr(A, B, Q, R)
    controller = LinearMPC(A, B, Q, R, horizon, -np.inf, np.inf, slew_cost=[[0.]])
    assert controller.terminal_cost_source == "dare"
    np.testing.assert_allclose(controller.terminal_cost, P)
    for x in [np.array([1., -.3]), np.array([-.2, 1.4])]:
        result = controller.solve(x)
        assert result.success, result.status
        np.testing.assert_allclose(result.u, -K @ x, atol=2e-7)
        np.testing.assert_allclose(result.predicted_x[0], x, atol=1e-8)
        np.testing.assert_allclose(result.predicted_x[1:],
                                   result.predicted_x[:-1] @ A.T + result.predicted_u @ B.T,
                                   atol=1e-8)
        assert result.predicted_x.shape == (horizon + 1, 2)
        assert result.predicted_u.shape == (horizon, 1)
        assert result.solve_time_s > 0
        assert result.iterations > 0
        assert result.max_constraint_violation <= controller.constraint_tolerance


def test_constant_equilibrium_reference_matches_shifted_lqr():
    A, B, Q, R = np.array([[.8]]), np.array([[.5]]), np.array([[2.]]), np.array([[.4]])
    K, _ = lqr(A, B, Q, R)
    controller = LinearMPC(A, B, Q, R, 4, -10, 10)
    x_ref, u_ref, x = np.array([2.]), np.array([.8]), np.array([1.3])
    result = controller.solve(x, x_ref=x_ref, u_ref=u_ref)
    assert result.success
    np.testing.assert_allclose(result.u, u_ref - K @ (x - x_ref), atol=1e-7)


def test_saturated_commands_are_bounded():
    controller = scalar_mpc(u_min=-.2, u_max=.3)
    for state, expected in [(20., -.2), (-20., .3)]:
        result = controller.solve([state])
        assert result.success, result.status
        np.testing.assert_allclose(result.u, [expected], atol=1e-7)
        assert np.all(result.predicted_u >= controller.u_min)
        assert np.all(result.predicted_u <= controller.u_max)
        assert result.max_constraint_violation < 1e-6


def test_slew_cost_uses_previous_command_and_reference():
    controller = scalar_mpc(horizon=1, Q=[[0.]], R=[[2.]], terminal_cost=[[0.]],
                            slew_cost=[[3.]], u_min=-10, u_max=10)
    for previous in [-1., .7]:
        result = controller.solve([3.], u_ref=[.4], u_prev=[previous])
        assert result.success
        np.testing.assert_allclose(result.u, [(2 * .4 + 3 * previous) / 5], atol=1e-8)
    result = controller.solve([3.], u_ref=[.4])
    np.testing.assert_allclose(result.u, [.4], atol=1e-8)


def test_slew_cost_penalizes_all_adjacent_horizon_commands():
    N, R, S, previous, reference = 4, 2., 3., -.7, .5
    controller = scalar_mpc(horizon=N, Q=[[0.]], R=[[R]], terminal_cost=[[0.]],
                            slew_cost=[[S]], u_min=-10, u_max=10)
    difference = np.eye(N) - np.eye(N, k=-1)
    rhs = np.full(N, R * reference)
    rhs[0] += S * previous
    expected = np.linalg.solve(R * np.eye(N) + S * difference.T @ difference, rhs)
    result = controller.solve([0.], u_ref=[reference], u_prev=[previous])
    assert result.success
    np.testing.assert_allclose(result.predicted_u[:, 0], expected, atol=1e-8)


def test_four_inputs_with_delay_and_short_horizon():
    A, B = augment_input_delay(.9 * np.eye(4), .1 * np.eye(4), 2)
    Q, R = np.diag([1.] * 4 + [.01] * 8), .1 * np.eye(4)
    controller = LinearMPC(A, B, Q, R, 1, [-.1, -.2, -.3, -.4], [.2, .3, .4, .5])
    result = controller.solve(np.arange(12) / 10, u_ref=np.zeros(4), u_prev=np.zeros(4))
    assert result.success, result.status
    assert result.u.shape == (4,)
    assert result.predicted_x.shape == (2, 12)
    assert result.predicted_u.shape == (1, 4)
    assert np.all(result.u >= controller.u_min)
    assert np.all(result.u <= controller.u_max)
    np.testing.assert_allclose(result.predicted_x[1], A @ result.predicted_x[0] + B @ result.u,
                               atol=1e-8)


def test_cached_solver_and_matrices_are_reused():
    controller = scalar_mpc()
    solver, P, constraints = controller._solver, controller._P, controller._constraints
    first = controller.solve([1.])
    second = controller.solve([-1.])
    assert first.success and second.success
    np.testing.assert_allclose(first.u, -second.u, atol=1e-8)
    assert controller._solver is solver
    assert controller._P is P
    assert controller._constraints is constraints


def test_dare_fallback_and_explicit_terminal_cost():
    with pytest.warns(RuntimeWarning, match="DARE terminal cost unavailable"):
        controller = scalar_mpc(A=[[1.1]], B=[[0.]])
    assert controller.terminal_cost_source == "stage"
    np.testing.assert_array_equal(controller.terminal_cost, controller.Q)
    assert controller.solve([1.]).success
    explicit = scalar_mpc(terminal_cost=[[0.]])
    assert explicit.terminal_cost_source == "provided"
    np.testing.assert_array_equal(explicit.terminal_cost, [[0.]])


@pytest.mark.parametrize("overrides", [
    {"A": [[1., 0.]]}, {"A": [[np.nan]]}, {"A": np.empty((0, 0))},
    {"B": [1.]}, {"B": [[1.], [2.]]}, {"B": np.empty((1, 0))}, {"B": [[np.inf]]},
    {"Q": [[-1.]]}, {"Q": [1.]}, {"Q": [[np.nan]]},
    {"R": [[0.]]}, {"R": [[-1.]]}, {"R": [[np.inf]]},
    {"horizon": 0}, {"horizon": -1}, {"horizon": 1.5}, {"horizon": True},
    {"u_min": 2., "u_max": 1.}, {"u_min": np.nan}, {"u_max": [1., 2.]},
    {"u_min": np.inf, "u_max": np.inf}, {"u_min": -np.inf, "u_max": -np.inf},
    {"terminal_cost": [[-1.]]}, {"terminal_cost": [[np.inf]]},
    {"slew_cost": [[-1.]]}, {"slew_cost": [[1., 0.], [0., 1.]]},
])
def test_invalid_constructor_data(overrides):
    with pytest.raises(ValueError):
        scalar_mpc(**overrides)


def test_nonsymmetric_weight_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        LinearMPC(np.eye(2), np.eye(2), [[1., .1], [0., 1.]], np.eye(2), 2, -1, 1)


@pytest.mark.parametrize("dt", [0., -.1, np.nan, np.inf, [.1]])
def test_invalid_sample_time(dt):
    with pytest.raises(ValueError):
        discretize([[0.]], [[1.]], dt)


@pytest.mark.parametrize("delay", [-1, 1.5, True])
def test_invalid_delay(delay):
    with pytest.raises(ValueError):
        augment_input_delay([[1.]], [[1.]], delay)


@pytest.mark.parametrize("kwargs", [
    {"x": [np.nan]}, {"x": [np.inf]}, {"x": [0., 0.]}, {"x": [[0.]]},
    {"x_ref": [np.inf]}, {"x_ref": [[0.], [1.]]},
    {"u_ref": [np.nan]}, {"u_ref": [0., 1.]},
    {"u_prev": [-np.inf]}, {"u_prev": []},
])
def test_invalid_solve_data(kwargs):
    controller = scalar_mpc()
    with pytest.raises(ValueError):
        controller.solve(**({"x": [0.]} | kwargs))
    assert controller.solve([0.]).success


@pytest.mark.parametrize("status,candidate", [
    ("maximum iterations reached", np.zeros(7)),
    ("primal infeasible", None),
    ("solved", np.full(7, np.nan)),
    ("solved", np.array([1e308, -1e308, 1e308, -1e308, 0., 0., 0.])),
    ("solved", np.array([0., 2., 4., 6., 2., 2., 2.])),
    ("solved inaccurate", np.array([0., .01, .02, .03, 0., 0., 0.])),
    ("solved", np.zeros(3)),
])
def test_solver_failure_returns_bounded_previous_input(monkeypatch, status, candidate):
    controller = scalar_mpc()
    fake = SimpleNamespace(info=SimpleNamespace(status=status, iter=23), x=candidate)
    monkeypatch.setattr(controller._solver, "solve", lambda **kwargs: fake)
    result = controller.solve([0.], u_ref=[-.2], u_prev=[3.])
    assert not result.success
    np.testing.assert_array_equal(result.u, [1.])
    assert np.all(np.isfinite(result.u))
    assert result.predicted_x.shape == (0, 1)
    assert result.predicted_u.shape == (0, 1)
    assert result.iterations == 23
    assert result.solve_time_s > 0
    if status.startswith("solved") and candidate.shape == (7,) and np.all(np.isfinite(candidate)):
        assert result.max_constraint_violation > controller.constraint_tolerance
        assert "constraint violation" in result.status


@pytest.mark.parametrize("method", ["update", "solve"])
def test_solver_exception_returns_reference_fallback(monkeypatch, method):
    controller = scalar_mpc()

    def fail(**kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(controller._solver, method, fail)
    result = controller.solve([0.], u_ref=[-.4])
    assert not result.success
    np.testing.assert_array_equal(result.u, [-.4])
    assert "forced failure" in result.status
    assert np.isinf(result.max_constraint_violation)


def test_only_numerical_bound_violation_is_projected(monkeypatch):
    controller = scalar_mpc(horizon=1, terminal_cost=[[0.]])
    candidate = np.array([0., 1. + 1e-9, 1. + 1e-9])
    fake = SimpleNamespace(info=SimpleNamespace(status="solved", iter=4), x=candidate)
    monkeypatch.setattr(controller._solver, "solve", lambda **kwargs: fake)
    result = controller.solve([0.])
    assert result.success
    np.testing.assert_array_equal(result.u, [1.])
    assert 0 < result.max_constraint_violation <= controller.constraint_tolerance


def test_equality_input_bounds():
    controller = scalar_mpc(u_min=.2, u_max=.2)
    result = controller.solve([1.])
    assert result.success
    np.testing.assert_array_equal(result.predicted_u, np.full((3, 1), .2))
