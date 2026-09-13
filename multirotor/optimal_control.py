from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import osqp
from scipy import linalg, sparse


FloatArray = NDArray[np.float64]


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _system(A: ArrayLike, B: ArrayLike) -> tuple[FloatArray, FloatArray]:
    A, B = _finite_array(A, "A"), _finite_array(B, "B")
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a nonempty square matrix")
    if B.ndim != 2 or B.shape[0] != A.shape[0] or B.shape[1] == 0:
        raise ValueError("B must have shape (number of states, number of inputs)")
    return A, B


def _weight(value: ArrayLike, size: int, name: str, positive: bool = False) -> FloatArray:
    matrix = _finite_array(value, name)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) / 2
    eigenvalues = linalg.eigvalsh(matrix)
    if (positive and eigenvalues[0] <= 0) or eigenvalues[0] < -1e-12:
        condition = "positive definite" if positive else "positive semidefinite"
        raise ValueError(f"{name} must be {condition}")
    return matrix


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    vector = _finite_array(value, name)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return vector


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def discretize(Ac: ArrayLike, Bc: ArrayLike, dt: float) -> tuple[FloatArray, FloatArray]:
    Ac, Bc = _system(Ac, Bc)
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    n, m = Bc.shape
    generator = np.zeros((n + m, n + m))
    generator[:n, :n], generator[:n, n:] = Ac, Bc
    transition = linalg.expm(generator * dt)
    if not np.all(np.isfinite(transition)):
        raise ValueError("discretization produced nonfinite matrices")
    return transition[:n, :n], transition[:n, n:]


def augment_input_delay(A: ArrayLike, B: ArrayLike, delay_steps: int) -> tuple[FloatArray, FloatArray]:
    A, B = _system(A, B)
    delay_steps = _integer(delay_steps, "delay_steps", 0)
    if delay_steps == 0:
        return A, B
    n, m = B.shape
    augmented_A = np.zeros((n + m * delay_steps, n + m * delay_steps))
    augmented_B = np.zeros((n + m * delay_steps, m))
    augmented_A[:n, :n], augmented_A[:n, n:n + m] = A, B
    augmented_A[n:-m, n + m:] = np.eye(m * (delay_steps - 1))
    augmented_B[-m:] = np.eye(m)
    return augmented_A, augmented_B


def lqr(A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike) -> tuple[FloatArray, FloatArray]:
    A, B = _system(A, B)
    Q = _weight(Q, A.shape[0], "Q")
    R = _weight(R, B.shape[1], "R", positive=True)
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = linalg.solve(R + B.T @ P @ B, B.T @ P @ A, assume_a="pos")
    return K, P


@dataclass
class MPCResult:
    u: FloatArray
    predicted_x: FloatArray
    predicted_u: FloatArray
    status: str
    solve_time_s: float
    iterations: int
    success: bool
    max_constraint_violation: float


class LinearMPC:
    constraint_tolerance = 1e-6

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        horizon: int,
        u_min: ArrayLike,
        u_max: ArrayLike,
        terminal_cost: ArrayLike | None = None,
        slew_cost: ArrayLike | None = None,
    ):
        self.A, self.B = _system(A, B)
        self.n, self.m = self.B.shape
        self.horizon = _integer(horizon, "horizon", 1)
        self.Q = _weight(Q, self.n, "Q")
        self.R = _weight(R, self.m, "R", positive=True)
        self.S = _weight(np.zeros((self.m, self.m)) if slew_cost is None else slew_cost,
                         self.m, "slew_cost")
        self.u_min = self._bound(u_min, "u_min")
        self.u_max = self._bound(u_max, "u_max")
        if np.any(self.u_min > self.u_max) or np.any(np.isposinf(self.u_min)) or np.any(np.isneginf(self.u_max)):
            raise ValueError("input bounds must define nonempty intervals containing finite values")
        if terminal_cost is None:
            try:
                _, self.terminal_cost = lqr(self.A, self.B, self.Q, self.R)
                self.terminal_cost_source = "dare"
            except (linalg.LinAlgError, ValueError):
                self.terminal_cost = self.Q.copy()
                self.terminal_cost_source = "stage"
                warnings.warn("DARE terminal cost unavailable; using Q", RuntimeWarning, stacklevel=2)
        else:
            self.terminal_cost = _weight(terminal_cost, self.n, "terminal_cost")
            self.terminal_cost_source = "provided"
        N, n, m = self.horizon, self.n, self.m
        self._state_size = (N + 1) * n
        state_cost = sparse.block_diag([sparse.csc_matrix(self.Q)] * N +
                                       [sparse.csc_matrix(self.terminal_cost)], format="csc")
        difference = sparse.eye(N, format="csc") - sparse.eye(N, k=-1, format="csc")
        difference = sparse.kron(difference, sparse.eye(m), format="csc")
        input_cost = sparse.kron(sparse.eye(N), self.R, format="csc")
        input_cost += difference.T @ sparse.kron(sparse.eye(N), self.S) @ difference
        self._P = sparse.triu(2 * sparse.block_diag([state_cost, input_cost]), format="csc")
        state_dynamics = sparse.eye((N + 1) * n, format="csc")
        state_dynamics -= sparse.kron(sparse.eye(N + 1, k=-1), self.A, format="csc")
        input_dynamics = sparse.vstack([
            sparse.csc_matrix((n, N * m)),
            sparse.kron(sparse.eye(N), -self.B),
        ], format="csc")
        dynamics = sparse.hstack([state_dynamics, input_dynamics], format="csc")
        bounds = sparse.hstack([
            sparse.csc_matrix((N * m, (N + 1) * n)), sparse.eye(N * m),
        ], format="csc")
        self._constraints = sparse.vstack([dynamics, bounds], format="csc")
        self._lower = np.concatenate([np.zeros((N + 1) * n), np.tile(self.u_min, N)])
        self._upper = np.concatenate([np.zeros((N + 1) * n), np.tile(self.u_max, N)])
        self._solver = osqp.OSQP()
        self._solver.setup(P=self._P, q=np.zeros((N + 1) * n + N * m),
                           A=self._constraints, l=self._lower, u=self._upper,
                           verbose=False, eps_abs=1e-8, eps_rel=1e-8,
                           max_iter=10000, polishing=True)

    def _bound(self, value: ArrayLike, name: str) -> FloatArray:
        bound = np.array(value, dtype=float, copy=True)
        if bound.ndim == 0:
            bound = np.full(self.m, bound.item())
        if bound.shape != (self.m,) or np.any(np.isnan(bound)):
            raise ValueError(f"{name} must be a scalar or shape ({self.m},), without NaNs")
        return bound

    def _constraint_violation(self, candidate: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            values = self._constraints @ candidate
            violation = max(0., np.max(self._lower - values), np.max(values - self._upper))
        return float(violation) if np.all(np.isfinite(values)) else float("inf")

    def solve(
        self,
        x: ArrayLike,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
        u_prev: ArrayLike | None = None,
    ) -> MPCResult:
        x = _vector(x, self.n, "x")
        x_ref = _vector(np.zeros(self.n) if x_ref is None else x_ref, self.n, "x_ref")
        u_ref = _vector(np.zeros(self.m) if u_ref is None else u_ref, self.m, "u_ref")
        u_prev = _vector(u_ref if u_prev is None else u_prev, self.m, "u_prev")
        start = perf_counter()
        fallback = np.clip(u_prev, self.u_min, self.u_max)
        q = np.concatenate([
            np.tile(-2 * self.Q @ x_ref, self.horizon),
            -2 * self.terminal_cost @ x_ref,
            np.tile(-2 * self.R @ u_ref, self.horizon),
        ])
        q[self._state_size:self._state_size + self.m] -= 2 * self.S @ u_prev
        if not np.all(np.isfinite(q)):
            raise ValueError("reference cost produced nonfinite linear terms")
        self._lower[:self.n] = self._upper[:self.n] = x
        iterations, violation = 0, float("inf")
        try:
            self._solver.update(q=q, l=self._lower, u=self._upper)
            result = self._solver.solve(raise_error=False)
            status, iterations = str(result.info.status), int(result.info.iter)
            candidate = result.x
            if candidate is not None and candidate.shape == q.shape and np.all(np.isfinite(candidate)):
                violation = self._constraint_violation(candidate)
                if status.lower() in ("solved", "solved inaccurate") and violation <= self.constraint_tolerance:
                    predicted_x = candidate[:self._state_size].reshape(self.horizon + 1, self.n).copy()
                    predicted_u = np.clip(candidate[self._state_size:].reshape(self.horizon, self.m),
                                          self.u_min, self.u_max)
                    projected = np.concatenate([predicted_x.ravel(), predicted_u.ravel()])
                    violation = max(violation, self._constraint_violation(projected))
                    if violation <= self.constraint_tolerance:
                        return MPCResult(predicted_u[0].copy(), predicted_x, predicted_u, status,
                                         perf_counter() - start, iterations, True, violation)
                if status.lower() in ("solved", "solved inaccurate"):
                    status += "; constraint violation exceeds tolerance"
            elif status.lower() in ("solved", "solved inaccurate"):
                status += "; invalid solution"
        except Exception as exc:
            status = f"solver error: {type(exc).__name__}: {exc}"
        return MPCResult(fallback, np.empty((0, self.n)), np.empty((0, self.m)), status,
                         perf_counter() - start, iterations, False, violation)
