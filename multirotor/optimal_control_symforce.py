"""SymForce copy of optimal_control.py, retargeted at codegen for firmware.

optimal_control.py solves the condensed MPC QP with OSQP, which only runs on
the host.  This module rewrites the same math on SymForce so every piece can
be emitted as dependency-free templated C++ and linked into the flight
control loop (exercised through the SimITL controller bridge, which executes
the real Betaflight firmware image).

Contents:

  * A symbolic mirror of four_motor_plant's dynamics (JAX -> symengine):
    battery sag, motor torque, prop thrust, substepped RK4, the ESC delay
    queue.  ``linearize`` differentiates the *discrete* map, so the emitted
    A/B match the nonlinear ``step`` exactly (unlike the expm path in
    four_motor_plant.discrete_model, which they agree with to RK4 accuracy).

  * ``RiccatiMPC``: drop-in replacement for ``LinearMPC`` with the same
    constructor and ``solve`` signature.  It solves the identical QP --
    tracking cost, slew penalty, input bounds -- by an exact backward
    Riccati recursion on the slew-augmented state z = [x; u_prev] followed
    by a forward rollout that clips each stage command to the input bounds.
    With no bound active the result equals OSQP's to solver tolerance.

  * ``generate``: emits the C++ consumed by the handwritten driver in
    multirotor/symforce_mpc (riccati_step/forward_step keep the driver a
    thin fixed-size loop), plus mpc_constants.h with the realized quad's
    trim/scale/weight/bounds constants.
"""

import tempfile
import typing as T
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

import symforce

symforce.set_epsilon_to_number()
import symforce.symbolic as sf  # noqa: E402

from symforce.codegen import Codegen  # noqa: E402
from symforce.codegen.backends.cpp.cpp_config import CppConfig  # noqa: E402
from symforce.codegen.backends.python.python_config import PythonConfig  # noqa: E402
import symforce.codegen.codegen_util as codegen_util  # noqa: E402

from multirotor import rate_plant as rp  # noqa: E402


FloatArray = NDArray[np.float64]

KT_NUMERATOR = rp.KT_NUMERATOR
RPM2RADS = float(rp.RPM2RADS)
_OCV_DRAWN = np.asarray(rp._OCV_DRAWN, dtype=float)
_OCV_VOLTS = np.asarray(rp._OCV_VOLTS, dtype=float)
SIM_TO_BF = np.asarray([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
DRAG_SIGNS = (-1., 1., -1.)
DRAG_COEFFS = (1e-5, 2e-6, 1e-5)
DRAG_OFFSET = 1e-6
DRAG_CLIP = 2.0

GENERATED_DIR = Path(__file__).resolve().parent / "symforce_mpc" / "generated"


# --- validation helpers (verbatim copy of optimal_control.py's) -------------


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


# --- numeric fallbacks kept from the original (lqr, delay augmentation) -----


def lqr(A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike) -> tuple[FloatArray, FloatArray]:
    A, B = _system(A, B)
    Q = _weight(Q, A.shape[0], "Q")
    R = _weight(R, B.shape[1], "R", positive=True)
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = linalg.solve(R + B.T @ P @ B, B.T @ P @ A, assume_a="pos")
    return K, P


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


# --- symbolic four-motor plant (mirror of four_motor_plant.py) --------------


def _clip(value: sf.Scalar, lo: float, hi: float) -> sf.Scalar:
    return sf.Min(sf.Max(value, lo), hi)


def _clip_vec(vec: sf.Matrix, lo: float, hi: float) -> sf.Matrix:
    return sf.Matrix([_clip(v, lo, hi) for v in vec.to_flat_list()])


def _ocv_per_cell(drawn: sf.Scalar) -> sf.Scalar:
    """Piecewise-linear OCV/SOC curve, clamped at both ends (jnp.interp)."""
    terms = [(float(_OCV_VOLTS[0]), drawn <= _OCV_DRAWN[0])]
    for i in range(1, len(_OCV_DRAWN)):
        d0, v0 = float(_OCV_DRAWN[i - 1]), float(_OCV_VOLTS[i - 1])
        d1, v1 = float(_OCV_DRAWN[i]), float(_OCV_VOLTS[i])
        terms.append((v0 + (v1 - v0) / (d1 - d0) * (drawn - d0), drawn <= d1))
    terms.append((float(_OCV_VOLTS[-1]), True))
    return sf.Piecewise(*terms)


def _battery_voltage(remaining_mah: sf.Scalar, motor_pwm: sf.Matrix, unit: dict) -> sf.Scalar:
    drawn = 1.0 - remaining_mah / max(unit["battery_capacity_mah"], 1.0)
    ocv = sf.Max(_ocv_per_cell(drawn) * unit["battery_cell_count"], 0.1)
    charge_inv = 1.0 - remaining_mah / max(unit["battery_capacity_charged_mah"], 1.0)
    mean_pwm = sf.Max(sum(_clip_vec(motor_pwm, 0., 1.).to_flat_list()) / 4.0, 0.0)
    sag = unit["max_voltage_sag"] * mean_pwm ** 2 * (1.0 + charge_inv ** 2)
    return _clip(ocv - sag, 0.0, 100.0)


def _motor_torque(volts: sf.Scalar, rpm: sf.Scalar, kv: float, unit: dict) -> sf.Scalar:
    kt = KT_NUMERATOR / max(kv, 0.0001)
    raw = (volts - rpm / kv) / max(unit["resistance"], 0.0001) * kt
    friction = unit["i0"] * kt
    return sf.Piecewise((raw - friction, rpm > 1.0),
                        (raw + friction, rpm < -1.0),
                        (raw - _clip(raw, -friction, friction), True))


def _prop_thrust(rpm: sf.Scalar, unit: dict) -> sf.Scalar:
    """Zero-inflow SimITL prop thrust b*rpm + a*rpm^2, clipped at 0."""
    max_rpm = max(unit["prop_max_rpm"], 0.01)
    prop_b = (unit["thrust_factor_z"] - unit["prop_a_factor"] * max_rpm ** 2) / max_rpm
    return sf.Max(prop_b * rpm + unit["prop_a_factor"] * rpm ** 2, 0.0)


def _physical_dynamics(x: sf.Matrix, sag_command: sf.Matrix, filter_command: sf.Matrix,
                       p: "SymbolicPlant") -> sf.Matrix:
    """Symbolic mirror of FourMotorPlant._physical_dynamics (11 states)."""
    xf = x.to_flat_list()
    filtered = _clip_vec(sf.Matrix(xf[7:11]), 0., 1.).to_flat_list()
    rpm = [w / RPM2RADS for w in xf[3:7]]
    vbat = _battery_voltage(p.remaining_mah_sym, sag_command, p.unit)
    torque, thrust = [], []
    for i in range(4):
        volts = sf.Max(vbat, 1.0) * filtered[i]
        torque.append(_motor_torque(volts, rpm[i], p.kv[i], p.unit))
        thrust.append(_prop_thrust(rpm[i], p.unit))
    tqf, inertia = p.unit["prop_torque_factor"], p.prop_inertia
    rotor_acc = [(t - tqf * th) / inertia for t, th in zip(torque, thrust)]
    rotor_acc = [sf.Piecewise((sf.Max(a, 0.), w <= 0.), (a, True))
                 for a, w in zip(rotor_acc, xf[3:7])]
    sim_rates = p.sim_to_bf.T * sf.Matrix(xf[0:3])
    moment = sum((sf.V3(*p.motor_positions[i]).cross(sf.V3(0., th, 0.))
                  for i, th in enumerate(thrust)), sf.V3(0., 0., 0.))
    moment[1] += sum(d * t for d, t in zip(p.motor_directions, torque))
    sr = sim_rates.to_flat_list()
    drag = [sr[a] * DRAG_SIGNS[a] * (sf.Abs(sr[a]) * DRAG_COEFFS[a]
            * p.frame_drag_area[a] + DRAG_OFFSET) for a in range(3)]
    moment += sf.V3(*[_clip(d, -DRAG_CLIP, DRAG_CLIP) for d in drag])
    sim_inertia = p.sim_inertia
    gyroscopic = sim_rates.cross(sim_inertia.multiply_elementwise(sim_rates))
    sim_acc = (moment - gyroscopic).multiply_elementwise(
        sf.V3(1. / sim_inertia[0], 1. / sim_inertia[1], 1. / sim_inertia[2]))
    rates_acc = p.sim_to_bf * sim_acc
    filter_acc = [p.esc_pole * (_clip(u, 0., 1.) - f)
                  for u, f in zip(filter_command.to_flat_list(), filtered)]
    return sf.Matrix(rates_acc.to_flat_list() + rotor_acc + filter_acc)


def _collective_thrust(x: sf.Matrix, p: "SymbolicPlant") -> sf.Scalar:
    return sum((_prop_thrust(w / RPM2RADS, p.unit) for w in x.to_flat_list()[3:7]), 0.0)


def _substep(physical: sf.Matrix, delayed: sf.Matrix, sag: sf.Matrix,
             p: "SymbolicPlant") -> sf.Matrix:
    """One RK4 substep of the physical state (mirror of _step's integrate)."""
    k1 = _physical_dynamics(physical, sag, delayed, p)
    k2 = _physical_dynamics(physical + 0.5 * p.h * k1, sag, delayed, p)
    k3 = _physical_dynamics(physical + 0.5 * p.h * k2, sag, delayed, p)
    k4 = _physical_dynamics(physical + p.h * k3, sag, delayed, p)
    physical = physical + p.h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    pf = physical.to_flat_list()
    pf[3:7] = [sf.Max(w, 0.) for w in pf[3:7]]
    pf[7:11] = [_clip(v, 0., 1.) for v in pf[7:11]]
    return sf.Matrix(pf)


class SymbolicPlant:
    """SymForce constants + symbolic builders for one FourMotorPlant."""

    def __init__(self, plant):
        unit = plant.unit
        self.plant = plant
        self.unit = unit
        self.dt = plant.dt
        self.delay_steps = plant.delay_steps
        self.substeps = plant.substeps
        self.h = plant.dt / plant.substeps
        self.state_size = plant.state_size
        self.esc_pole = plant.esc_pole
        self.prop_inertia = plant.prop_inertia
        self.kv = [float(unit["kv"]) * float(v) for v in np.asarray(plant.motor_variance)]
        self.sim_inertia = sf.V3(*np.asarray(unit["inertia"], float) * [1., -1., 1.])
        self.sim_to_bf = sf.Matrix(SIM_TO_BF.tolist())
        self.motor_positions = np.asarray(unit["motor_positions"], float)
        self.motor_directions = np.asarray(unit.get("motor_directions", [-1., 1., 1., -1.]), float)
        self.frame_drag_area = np.asarray(unit.get("frame_drag_area", np.zeros(3)), float)
        self.remaining_mah_sym = sf.Symbol("remaining_mah")

    def symbols(self):
        n = self.state_size
        x = sf.Matrix([sf.Symbol(f"x{i}") for i in range(n)])
        u = sf.Matrix([sf.Symbol(f"u{i}") for i in range(4)])
        return x, u, self.remaining_mah_sym

    def substep(self, physical, delayed, sag):
        return _substep(physical, delayed, sag, self)

    def collective_thrust(self, x):
        return _collective_thrust(x, self)


# --- generated-function evaluation (Python backend) -------------------------


def _matrix_type(rows: int, cols: int):
    return type(sf.Matrix(np.zeros((rows, cols))))


def _substep_wrapper(p: SymbolicPlant):
    T_x, T_u = _matrix_type(11, 1), _matrix_type(4, 1)

    def substep(x: T_x, delayed: T_u, sag: T_u,
                remaining_mah: sf.Scalar) -> T_x:
        return p.substep(x, delayed, sag)

    return substep


def _collective_wrapper(p: SymbolicPlant):
    T_x = _matrix_type(11, 1)

    def collective_thrust(x: T_x, remaining_mah: sf.Scalar) -> sf.Scalar:
        return p.collective_thrust(x)

    return collective_thrust


def _plant_step_wrapper(p: SymbolicPlant):
    """Full control-period step (substeps + queue shift) for host-side sims."""
    T_x, T_u = _matrix_type(p.state_size, 1), _matrix_type(4, 1)

    def plant_step(x: T_x, u: T_u, remaining_mah: sf.Scalar) -> T_x:
        uf = _clip_vec(u, 0., 1.).to_flat_list()
        delayed = sf.Matrix(x.to_flat_list()[11:15]) if p.delay_steps else u
        sag = sf.Matrix(x.to_flat_list()[-4:]) if p.delay_steps else u
        physical = sf.Matrix(x.to_flat_list()[:11])
        for _ in range(p.substeps):
            physical = p.substep(physical, delayed, sag)
        if not p.delay_steps:
            return physical
        tail = x.to_flat_list()[15:]
        shifted = sf.Matrix(tail + uf) if tail else sf.Matrix(uf)
        return physical.col_join(shifted)

    return plant_step


def _python_module(p: SymbolicPlant, cache_dir: Path | None = None):
    """Codegen the plant's Python functions once and import them."""
    key = (p.state_size, p.delay_steps, p.substeps, p.dt,
           float(p.unit["kv"]), float(p.unit["prop_inertia"]),
           float(p.unit["prop_max_rpm"]))
    cache_dir = Path(cache_dir or
                     Path(tempfile.gettempdir()) / f"sf_mpc_{abs(hash(key))}")
    names = ("sf_substep", "sf_substep_jac", "sf_collective_thrust_jac")
    if not all((cache_dir / f"{n}.py").is_file() for n in names):
        cache_dir.mkdir(parents=True, exist_ok=True)
        config = PythonConfig()
        sub = Codegen.function(_substep_wrapper(p), config=config,
                               name="sf_substep")
        jac = sub.with_jacobians(which_args=["x", "delayed", "sag"],
                                 name="sf_substep_jac")
        coll = Codegen.function(_collective_wrapper(p), config=config,
                                name="sf_collective_thrust")
        coll = coll.with_jacobians(which_args=["x"], name="sf_collective_thrust_jac")
        for cg in (sub, jac, coll):
            cg.generate_function(output_dir=str(cache_dir),
                                 skip_directory_nesting=True)
    funcs = {}
    for n in names:
        funcs[n] = codegen_util.load_generated_function(n, cache_dir)
    return funcs


def _col(v: ArrayLike) -> FloatArray:
    return np.asarray(v, dtype=float).reshape(-1, 1)


def _op_inputs(plant, x_op, u_op, p):
    """Split an operating point into (physical, delayed, sag, u) substep args."""
    x_op = plant.trim_state if x_op is None else np.asarray(x_op, float)
    u_op = plant.trim_command if u_op is None else np.asarray(u_op, float)
    u_op = np.clip(u_op, 0., 1.)
    delayed = x_op[11:15] if p.delay_steps else u_op
    sag = x_op[-4:] if p.delay_steps else u_op
    return x_op[:11], delayed, sag, u_op


def step(plant, x, u, remaining_mah=None, cache_dir=None):
    """Nonlinear full-state step, mirror of FourMotorPlant.step."""
    p = SymbolicPlant(plant)
    funcs = _python_module(p, cache_dir)
    x = _vector(x, p.state_size, "x")
    u = np.clip(_vector(u, 4, "u"), 0., 1.)
    mah = plant.remaining_mah if remaining_mah is None else float(remaining_mah)
    physical, delayed, sag, u = _op_inputs(plant, x, u, p)
    for _ in range(p.substeps):
        physical = np.asarray(
            funcs["sf_substep"](_col(physical), _col(delayed), _col(sag), mah)).ravel()
    if not p.delay_steps:
        return physical
    return np.concatenate([physical, x[15:], u])


def linearize(plant, x_op=None, u_op=None, remaining_mah=None, cache_dir=None):
    """Discrete (x_next, A, B) at an operating point via the symbolic RK4 map.

    Defaults to the plant's hover trim, matching FourMotorPlant.discrete_model
    (which linearizes the *continuous* map and discretizes by expm instead);
    the symbolic path differentiates the substepped RK4 the nonlinear step
    actually runs, and composes per-substep jacobians by the chain rule.
    """
    p = SymbolicPlant(plant)
    funcs = _python_module(p, cache_dir)
    mah = plant.remaining_mah if remaining_mah is None else float(remaining_mah)
    physical, delayed, sag, u_op = _op_inputs(plant, x_op, u_op, p)
    acc = np.zeros((11, 19))
    acc[:, :11] = np.eye(11)
    for _ in range(p.substeps):
        out = funcs["sf_substep_jac"](_col(physical), _col(delayed), _col(sag), mah)
        physical = np.asarray(out[0]).ravel()
        acc = np.asarray(out[1]) @ acc
        acc[:, 11:15] += np.asarray(out[2])
        acc[:, 15:19] += np.asarray(out[3])
    n, d = p.state_size, p.delay_steps
    x_op_full = plant.trim_state if x_op is None else np.asarray(x_op, float)
    x_next = np.concatenate([physical, x_op_full[15:],
                             np.clip(u_op, 0., 1.)]) if d else physical
    A = np.zeros((n, n))
    A[:11, :11] = acc[:, :11]
    if d:
        A[:11, 11:15] += acc[:, 11:15]
        A[:11, -4:] += acc[:, 15:19]
        A[11:-4, 15:] = np.eye(4 * (d - 1))
    B = np.zeros((n, 4))
    B[-4:] = np.eye(4)
    return x_next, A, B


def discrete_model(plant, **kwargs):
    """(A, B) matching FourMotorPlant.discrete_model, via symbolic jacobian."""
    _, A, B = linearize(plant, **kwargs)
    return A, B


# --- Riccati-recursion MPC (LinearMPC-compatible) ---------------------------
#
# Same QP as LinearMPC: stage cost  (x-xr)'Q(x-xr) + (u-ur)'R(u-ur)
#                       + (u_k - u_{k-1})'S(u_k - u_{k-1}),  u_{-1} = u_prev,
# terminal (x_N-xr)'Pf(x_N-xr), dynamics x' = Ax + Bu, bounds lo <= u <= hi.
#
# Augmenting with u_{k-1} gives the homogeneous LQ tracking problem on
# z = [x; u_prev]  (dim z = n+m):
#   Az = [[A,0],[0,0]]   Bz = [[B],[I]]
#   stage  = z'Qz z + 2 z'Nz u + u'Rz u + 2 z'c_z + 2 u'c_u
#   Qz = diag(Q,S)  Rz = R+S  Nz = [[0],[-S]]  c_z = [-Q xr; 0]  c_u = -R ur
# Backward recursion on the quadratic value function z'P z + 2 z'g:
#   G = Rz + Bz'P Bz        M = Bz'P Az + Nz'      h = Bz'g + c_u
#   K = G^-1 M              f = G^-1 h
#   P <- Qz + Az'P Az - M'G^-1 M        g <- c_z + Az'g - M'f
# Forward: u_k = clip(-(K_k z_k + f_k));  z_{k+1} = Az z_k + Bz u_k.
# riccati_step / forward_step in the generated C++ implement the identical
# recursions, so the embedded solver is bit-for-bit this class.


class RiccatiMPC:
    """Drop-in replacement for optimal_control.LinearMPC (same API)."""

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
        n, m = self.n, self.m
        self._z = n + m
        self.Az = np.block([[self.A, np.zeros((n, m))],
                            [np.zeros((m, n)), np.zeros((m, m))]])
        self.Bz = np.vstack([self.B, np.eye(m)])
        self.Qz = linalg.block_diag(self.Q, self.S)
        self.Rz = self.R + self.S
        self.Nz = np.vstack([np.zeros((n, m)), -self.S])
        self.Pz = linalg.block_diag(self.terminal_cost, np.zeros((m, m)))

    def _bound(self, value: ArrayLike, name: str) -> FloatArray:
        bound = np.array(value, dtype=float, copy=True)
        if bound.ndim == 0:
            bound = np.full(self.m, bound.item())
        if bound.shape != (self.m,) or np.any(np.isnan(bound)):
            raise ValueError(f"{name} must be a scalar or shape ({self.m},), without NaNs")
        return bound

    def _gains(self, x_ref: FloatArray, u_ref: FloatArray):
        """Backward pass: per-stage feedback K, feedforward f."""
        n, m, N = self.n, self.m, self.horizon
        c_z = np.concatenate([-self.Q @ x_ref, np.zeros(m)])
        c_u = -self.R @ u_ref
        P, g = self.Pz.copy(), np.concatenate([-self.terminal_cost @ x_ref, np.zeros(m)])
        gains = [None] * N
        for k in range(N - 1, -1, -1):
            G = self.Rz + self.Bz.T @ P @ self.Bz
            M = self.Bz.T @ P @ self.Az + self.Nz.T
            f = linalg.solve(G, self.Bz.T @ g + c_u, assume_a="pos")
            K = linalg.solve(G, M, assume_a="pos")
            P = self.Qz + self.Az.T @ P @ self.Az - M.T @ K
            P = (P + P.T) / 2
            g = c_z + self.Az.T @ g - M.T @ f
            gains[k] = (K, f)
        return gains

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
        try:
            gains = self._gains(x_ref, u_ref)
        except Exception as exc:
            return MPCResult(fallback, np.empty((0, self.n)), np.empty((0, self.m)),
                             f"solver error: {type(exc).__name__}: {exc}",
                             perf_counter() - start, 0, False, float("inf"))
        z = np.concatenate([x, u_prev])
        predicted_x, predicted_u = [x.copy()], []
        clip_violation = 0.0
        for K, f in gains:
            raw = -(K @ z + f)
            u = np.clip(raw, self.u_min, self.u_max)
            clip_violation = max(clip_violation, float(np.max(np.abs(raw - u))))
            predicted_u.append(u)
            z = self.Az @ z + self.Bz @ u
            predicted_x.append(z[:self.n].copy())
        status = "riccati" if clip_violation <= self.constraint_tolerance else \
            "riccati; input clip active"
        return MPCResult(predicted_u[0].copy(), np.asarray(predicted_x),
                         np.asarray(predicted_u), status, perf_counter() - start,
                         self.horizon, True, clip_violation)


# --- codegen -----------------------------------------------------------------


def _riccati_step_wrapper(n: int, m: int):
    """Symbolic backward pass; identical math to RiccatiMPC._gains."""
    z = n + m
    T_A, T_B = _matrix_type(n, n), _matrix_type(n, m)
    T_Q, T_R = _matrix_type(n, n), _matrix_type(m, m)
    T_xr, T_ur = _matrix_type(n, 1), _matrix_type(m, 1)
    T_P, T_g = _matrix_type(z, z), _matrix_type(z, 1)
    T_K, T_f = _matrix_type(m, z), _matrix_type(m, 1)

    def riccati_step(A: T_A, B: T_B, Q: T_Q, R: T_R, S: T_R,
                     x_ref: T_xr, u_ref: T_ur, P_next: T_P, g_next: T_g
                     ) -> T.Tuple[T_K, T_f, T_P, T_g]:
        zero_nm, zero_mm = sf.Matrix.zeros(n, m), sf.Matrix.zeros(m, m)
        Az = sf.Matrix.block_matrix([[A, zero_nm], [zero_nm.T, zero_mm]])
        Bz = sf.Matrix.block_matrix([[B], [sf.Matrix.eye(m)]])
        Qz = sf.Matrix.block_matrix([[Q, zero_nm], [zero_nm.T, S]])
        Rz = R + S
        Nz = sf.Matrix.block_matrix([[zero_nm], [-S]])
        c_z = sf.Matrix((-Q * x_ref).to_flat_list() + [0.] * m)
        c_u = -R * u_ref
        G = Rz + Bz.T * P_next * Bz
        M = Bz.T * P_next * Az + Nz.T
        Gi = G.inv()
        K, f = Gi * M, Gi * (Bz.T * g_next + c_u)
        P = Qz + Az.T * P_next * Az - M.T * K
        P = (P + P.T) * 0.5
        g = c_z + Az.T * g_next - M.T * f
        return K, f, P, g

    return riccati_step


def _forward_step_wrapper(n: int, m: int):
    """One forward-rollout stage: clip(-(Kz+f)), then z' = Az z + Bz u."""
    z = n + m
    T_z, T_K, T_f = _matrix_type(z, 1), _matrix_type(m, z), _matrix_type(m, 1)
    T_A, T_B, T_u = _matrix_type(n, n), _matrix_type(n, m), _matrix_type(m, 1)

    def forward_step(z_state: T_z, K: T_K, f: T_f, A: T_A, B: T_B,
                     u_min: T_u, u_max: T_u) -> T.Tuple[T_u, T_z]:
        zf = z_state.to_flat_list()
        u = sf.Matrix([_clip(v, u_min[i], u_max[i])
                       for i, v in enumerate((-(K * z_state + f)).to_flat_list())])
        x_next = A * sf.Matrix(zf[:n]) + B * u
        return u, sf.Matrix(x_next.to_flat_list() + u.to_flat_list())

    return forward_step


def controller_mats(p: SymbolicPlant, effort_weight: float,
                    collective_weight: float, rate_scale: float,
                    noise_dps: float = 1.):
    """Scaled A,B,Q,R, DARE terminal P and the 19x7 target-solve map.

    Mirrors FourMotorController's construction: scaled deviation coords,
    Q = dt*(C'C + collective_weight * c c') with c the scaled collective-
    thrust gradient, and target = M^-1 rhs for rhs = [D bias; ref; coll-1].
    The baked kTargetInv columns are [bias*dt (0..2), ref (3..5), coll (6)].
    """
    plant = p.plant
    n = p.state_size
    _, A, B = linearize(plant)
    scale = np.r_[np.full(3, rate_scale), plant.trim_state[3:7], np.ones(n - 7)]
    As = A * scale[None, :] / scale[:, None]
    Bs = B / scale[:, None]
    funcs = _python_module(p)
    trim_phys = _col(plant.trim_state[:11])
    dTdx = np.asarray(funcs["sf_collective_thrust_jac"](
        trim_phys, plant.remaining_mah)[1]).ravel()
    coll = np.zeros(n)
    coll[:11] = dTdx * scale[:11] / (plant.mass * 9.81)
    C = np.eye(3, n)
    Q = plant.dt * (C.T @ C + collective_weight * np.outer(coll, coll))
    R = np.eye(4) * plant.dt * effort_weight
    _, P = lqr(As, Bs, Q, R)
    target_matrix = np.block([
        [np.eye(n) - As, -Bs],
        [C, np.zeros((3, 4))],
        [coll[None, :], np.zeros((1, 4))],
    ])
    target_inv = np.linalg.inv(target_matrix)
    cols = np.zeros((n + 4, 7))
    cols[:, :3] = target_inv[:, :3] * plant.dt        # disturbance bias * dt
    cols[:, 3:6] = target_inv[:, n:n + 3]             # scaled rate reference
    cols[:, 6] = target_inv[:, n + 3]                 # collective - 1
    # Bias observer gain (same construction as RateStateObserver): DARE on the
    # bias-augmented system gives the Kalman gain; only the 3 bias rows are
    # needed since the physical state is measured, not estimated.
    C3 = np.eye(3, n)
    D = C3.T * plant.dt
    A_aug = np.block([[As, D], [np.zeros((3, n)), np.eye(3)]])
    C_aug = np.c_[C3, np.zeros((3, 3))]
    process = np.diag(np.r_[np.full(3, 2e-4), np.full(4, 1e-3),
                            np.full(n - 7, 1e-10), np.full(3, 10.)]) * plant.dt
    measurement = np.eye(3) * max((noise_dps / 100.)**2, 1e-8)
    P_obs = linalg.solve_discrete_are(A_aug.T, C_aug.T, process, measurement)
    L_full = np.linalg.solve(C_aug @ P_obs @ C_aug.T + measurement,
                             C_aug @ P_obs).T
    bias_gain = L_full[n:, :]                          # 3x3: innovation -> bias
    return As, Bs, Q, R, P, cols, bias_gain


def _constants_header(p: SymbolicPlant, horizon: int, effort_weight: float,
                      slew_weight: float, collective_weight: float,
                      rate_scale: float) -> str:
    plant = p.plant
    scale = np.r_[np.full(3, rate_scale), plant.trim_state[3:7],
                  np.ones(plant.state_size - 7)]
    u_dev_min, u_dev_max = 0.05 - plant.trim_command, 1.0 - plant.trim_command
    # noise_dps=3 slows the bias observer vs the python reference (1.0):
    # broadband propwash noise otherwise wanders the equilibrium target.
    _, _, Q, R, P, target_inv, bias_gain = controller_mats(
        p, effort_weight, collective_weight, rate_scale, noise_dps=3.)
    # Feasible collective cap: nonlinear equilibrium thrust at u=0.85 - leaves
    # ~0.15 pwm of differential headroom for rate authority at full throttle
    # (u=0.95-1.0 saturates the axes and spills rate commands cross-axis).
    import jax.numpy as jnp
    x_eq = np.asarray(plant.trim_state)
    for _ in range(2000):
        x_eq = np.asarray(plant.step(jnp.asarray(x_eq),
                                     jnp.asarray([0.85] * 4)))
    coll_max = float(plant.collective_thrust(jnp.asarray(x_eq))) / (
        plant.mass * 9.81)
    # Gain-schedule operating points: hover trim, mid, and near-saturation
    # equilibria (physical states; the driver relinearizes + retabulates gains
    # at each and interpolates by measured rotor speed).
    sched_u = [float(plant.trim_command[0]), 0.7, 0.92]
    sched_x, sched_w = [], []
    for u in sched_u:
        xs = np.asarray(plant.trim_state)
        for _ in range(2000):
            xs = np.asarray(plant.step(jnp.asarray(xs), jnp.asarray([u] * 4)))
        sched_x.append(xs[:11])
        sched_w.append(float(np.mean(xs[3:7])))

    def fmt(vec, per=8):
        vec = np.atleast_1d(np.asarray(vec, float))
        lines = [", ".join(f"{v:.17g}" for v in vec[i:i + per])
                 for i in range(0, len(vec), per)]
        return ",\n  ".join(lines)

    return f"""// Autogenerated by multirotor.optimal_control_symforce -- do not edit.
#pragma once
namespace quad_mpc {{
// Quad config this solver's constants were generated for (file stem).
constexpr const char* kQuadKey = "{rp.DEFAULT_QUAD_CONFIG.stem}";
constexpr int kStates = {p.state_size};
constexpr int kPhysicalStates = 11;
constexpr int kInputs = 4;
constexpr int kVIn = 3;  // differential inputs (roll, pitch, yaw)
// Target solve works on [x; u4] (19); the QP's slew-augmented state is
// [x; v_prev3] (18).
constexpr int kAugmented = {p.state_size + 4};
constexpr int kAugZ = {p.state_size + 3};
constexpr int kDelaySteps = {p.delay_steps};
constexpr int kSubsteps = {p.substeps};
constexpr int kHorizon = {horizon};
constexpr double kDt = {plant.dt:.17g};
constexpr double kRateScale = {rate_scale:.17g};
constexpr double kRpmToRadS = {RPM2RADS:.17g};
constexpr double kCollectiveWeight = {collective_weight:.17g};
constexpr double kEffortWeight = {effort_weight:.17g};
constexpr double kSlewWeight = {slew_weight:.17g};
constexpr double kMassTimesG = {plant.mass * 9.81:.17g};
constexpr double kRemainingMah = {plant.remaining_mah:.17g};
constexpr double kCommandFloor = 0.05;
constexpr double kTrimState[kStates] = {{
  {fmt(plant.trim_state)}
}};
constexpr double kTrimCommand[kInputs] = {{
  {fmt(plant.trim_command)}
}};
constexpr double kScale[kStates] = {{
  {fmt(scale)}
}};
constexpr double kUDevMin[kInputs] = {{
  {fmt(u_dev_min)}
}};
constexpr double kUDevMax[kInputs] = {{
  {fmt(u_dev_max)}
}};
// Q (scaled deviation coords) = dt*(C'C + kCollectiveWeight * c c')
constexpr double kQ[kStates * kStates] = {{
  {fmt(Q.ravel(order="F"))}
}};
// Mixer reparameterization: applied command u = m*1 + M*v, where m is the
// collective (mean) channel pinned to the target and v is the 3-dim
// differential the QP optimizes.  Columns: roll (-z sign), pitch (+x sign),
// yaw (motor_directions) - orthonormal up to scale 4, so M'*M = 4I and the
// v-space effort/slew weights are 4x the motor-space ones.
constexpr double kMixV[4 * 3] = {{
  {fmt(np.array([[ 1., 1., -1.],      # motor 0 (+x, -z, dir -1)
                [-1., 1.,  1.],      # motor 1 (+x, +z, dir +1)
                [ 1.,-1.,  1.],      # motor 2 (-x, -z, dir +1)
                [-1.,-1., -1.]]).ravel(order="C"))}
}};
// Effort/slew cost on v (isotropic; 4x because M'M = 4I).
constexpr double kRv = {4 * effort_weight * plant.dt:.17g};
constexpr double kSv = {4 * slew_weight * plant.dt:.17g};
// DARE terminal cost on the physical+queue states
constexpr double kTerminalCost[kStates * kStates] = {{
  {fmt(P.ravel(order="F"))}
}};
// target = kTargetInv[:, 3:6] @ (ref/kRateScale) + kTargetInv[:, 6]*(coll-1);
// cols 0..2 multiply the estimated disturbance bias.
constexpr double kTargetInv[kAugmented * 7] = {{
  {fmt(target_inv.ravel(order="F"))}
}};
// Rate-innovation -> per-step disturbance-bias gain (Kalman L, bias rows).
constexpr double kBiasGain[3 * 3] = {{
  {fmt(bias_gain.ravel(order="F"))}
}};
// bias units: scaled rate increment per second (innovation/dt); a badly
// damaged motor needs ~50, so clamp generously.
constexpr double kBiasMax = 500.;
// Collective thrust in Newtons at trim state (= mass * g by construction of
// trim). Used to refine the linearized collective target row.
constexpr double kHoverThrust = {p.plant.mass * 9.81:.17g};
// Throttle-stick mapping: hover stick position = trim command; full stick
// requests kCollMax (feasible collective at u=0.95, model units).
constexpr double kHoverStick = {p.plant.trim_command[0]:.17g};
constexpr double kCollMax = {coll_max:.17g};
constexpr int kNumSched = 3;
constexpr double kSchedW[3] = {{{', '.join(f'{w:.17g}' for w in sched_w)}}};
constexpr double kSchedX[3][11] = {{
{chr(10).join('  {' + ', '.join(f'{v:.17g}' for v in x) + '},' for x in sched_x)}
}};
constexpr double kSchedU[3][4] = {{
{chr(10).join('  {' + ', '.join(f'{u:.17g}' for _ in range(4)) + '},' for u in sched_u)}
}};
}}  // namespace quad_mpc
"""


def generate(output_dir: Path | str = GENERATED_DIR, plant=None,
             horizon_ms: float = 40., effort_weight: float = .1,
             slew_weight: float = 0., collective_weight: float = 10.,
             rate_scale_dps: float = 100.) -> Path:
    """Emit C++ (generated/*.h) + mpc_constants.h for the embedded solver."""
    if plant is None:
        from multirotor.four_motor_plant import FourMotorPlant
        plant = FourMotorPlant()
    p = SymbolicPlant(plant)
    horizon = int(round(horizon_ms / (plant.dt * 1000)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CppConfig()
    sub = Codegen.function(_substep_wrapper(p), config=config, name="substep")
    generated = [sub]
    generated.append(sub.with_jacobians(which_args=["x", "delayed", "sag"],
                                        name="substep_jac"))
    generated.append(Codegen.function(_plant_step_wrapper(p), config=config,
                                      name="plant_step"))
    generated.append(Codegen.function(_riccati_step_wrapper(p.state_size, 3),
                                      config=config, name="riccati_step",
                                      output_names=["K", "f", "P", "g"]))
    generated.append(Codegen.function(_forward_step_wrapper(p.state_size, 3),
                                      config=config, name="forward_step",
                                      output_names=["u", "z_next"]))
    coll = Codegen.function(_collective_wrapper(p), config=config,
                            name="collective_thrust")
    generated.append(coll)
    generated.append(coll.with_jacobians(which_args=["x"],
                                         name="collective_thrust_jac"))
    files = []
    for cg in generated:
        files += cg.generate_function(output_dir=str(output_dir),
                                      skip_directory_nesting=True).generated_files
    constants = output_dir / "mpc_constants.h"
    constants.write_text(_constants_header(p, horizon, effort_weight, slew_weight,
                                           collective_weight, np.deg2rad(rate_scale_dps)))
    files.append(constants)
    for f in files:
        print(f"  wrote {f}")
    return output_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=GENERATED_DIR)
    parser.add_argument("--horizon-ms", type=float, default=40.)
    parser.add_argument("--effort", type=float, default=.1)
    parser.add_argument("--slew", type=float, default=0.)
    parser.add_argument("--collective", type=float, default=10.)
    parser.add_argument("--check", action="store_true",
                        help="verify symbolic A/B vs FourMotorPlant.discrete_model")
    args = parser.parse_args()
    from multirotor.four_motor_plant import FourMotorPlant
    plant = FourMotorPlant()
    if args.check:
        _, A, B = linearize(plant)
        A_ref, B_ref = plant.discrete_model()
        err = np.max(np.abs(A - A_ref)), np.max(np.abs(B - B_ref))
        print(f"symbolic-vs-expm jacobian max|dA|={err[0]:.3e} max|dB|={err[1]:.3e}")
    out = generate(args.out, plant, args.horizon_ms, args.effort, args.slew,
                   args.collective)
    print(f"generated embedded MPC sources in {out}")


if __name__ == "__main__":
    main()
