"""Core flap-servo geometry solver using scipy numerical root finders.

Coordinate convention
---------------------
* Hinge / flap rotation axis is at the origin.
* Flap angle 0 points along the negative x-axis.
* Positive angles are measured clockwise from the negative x-axis.
* Servo endpoint moves on the horizontal line y = servo_y, from
  x = servo_x at input 0 to x = servo_x + servo_travel at input 1.
* The control-rod attachment point on the flap has local offsets
  (flap_x, flap_y).  flap_x is measured along the flap (positive in the
  trailing-edge direction when the angle is 0).  flap_y is measured
  perpendicular to the flap, positive clockwise-90 from flap_x.
"""

from typing import Union

import numpy as np
from scipy.optimize import brentq

FloatArray = Union[float, np.ndarray]


def _attach_point(theta: float, flap_x: float, flap_y: float):
    """Global coordinates of the rod attachment point for a given flap angle."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    px = -flap_x * cos_t + flap_y * sin_t
    py = flap_x * sin_t + flap_y * cos_t
    return float(px), float(py)


def _residual(theta, Sx: float, Sy: float, flap_x: float, flap_y: float, rod_length: float):
    """Squared distance from servo endpoint to attachment point minus L^2."""
    theta = np.asarray(theta, dtype=float)
    px = -flap_x * np.cos(theta) + flap_y * np.sin(theta)
    py = flap_x * np.sin(theta) + flap_y * np.cos(theta)
    return (Sx - px) ** 2 + (Sy - py) ** 2 - rod_length ** 2


def _find_roots(Sx: float, Sy: float, flap_x: float, flap_y: float, rod_length: float,
                angle_bounds=(-np.pi, np.pi), grid_points: int = 1000, seed: float = None):
    """Find all flap angles in the bounds that satisfy the rod-length constraint."""
    thetas = np.linspace(angle_bounds[0], angle_bounds[1], grid_points)
    r = _residual(thetas, Sx, Sy, flap_x, flap_y, rod_length)

    roots = set()
    tol = 1e-10
    # Exact hits on grid points
    for i in range(grid_points):
        if abs(r[i]) < tol:
            roots.add(float(thetas[i]))

    # Sign-change brackets
    for i in range(grid_points - 1):
        if np.isnan(r[i]) or np.isnan(r[i + 1]):
            continue
        if r[i] == 0.0 or r[i + 1] == 0.0:
            continue
        if r[i] * r[i + 1] < 0.0:
            try:
                root = brentq(
                    _residual,
                    float(thetas[i]),
                    float(thetas[i + 1]),
                    args=(Sx, Sy, flap_x, flap_y, rod_length),
                    xtol=1e-12,
                )
                roots.add(root)
            except ValueError:
                pass

    # Seed angle (e.g. the previous solution) may sit exactly on a root that the
    # grid missed, most commonly at a tangent (double) root.
    if seed is not None and angle_bounds[0] <= seed <= angle_bounds[1]:
        if abs(_residual(seed, Sx, Sy, flap_x, flap_y, rod_length)) < 1e-7:
            roots.add(float(seed))

    return sorted(roots)


def _pick_root(roots, prev: float):
    """Choose the root whose unwrapped value is closest to prev.

    If two roots tie in distance, prefer the one with positive unwrapped
    delta from prev.
    """
    if np.isnan(prev):
        return roots[0]

    def key(r):
        delta = ((r - prev + np.pi) % (2.0 * np.pi)) - np.pi
        return (abs(delta), -delta)

    return min(roots, key=key)


def _solve_theta(inputs: np.ndarray,
                 servo_x: float, servo_y: float, servo_travel: float,
                 flap_x: float, flap_y: float, rod_length: float,
                 initial_angle: float,
                 angle_bounds=(-np.pi, np.pi),
                 grid_points: int = 600) -> np.ndarray:
    n = inputs.size
    theta = np.full(n, np.nan)
    prev = float(initial_angle)

    for i, u in enumerate(inputs):
        Sx = servo_x + u * servo_travel
        x0 = float(prev) if not np.isnan(prev) else float(initial_angle)

        # Find all real roots in the bounded angle range and pick the one that
        # keeps the solution on the same unwrapped branch as the previous input.
        roots = _find_roots(
            Sx, servo_y, flap_x, flap_y, rod_length,
            angle_bounds=angle_bounds, grid_points=grid_points,
            seed=x0,
        )
        if not roots:
            prev = np.nan
            continue

        candidate = _pick_root(roots, prev)
        if not np.isnan(prev):
            delta = ((candidate - prev + np.pi) % (2.0 * np.pi)) - np.pi
            candidate = prev + delta

        theta[i] = candidate
        prev = candidate

    return theta


def _endpoint_angle_sum(rod_length: float,
                        servo_x: float, servo_y: float, servo_travel: float,
                        flap_x: float, flap_y: float,
                        grid_points: int = 200) -> float:
    """Return θ(0) + θ(1) for the continuous branch; inf if no real solution."""
    res = simulate_flap([0.0, 1.0], servo_x, servo_y, servo_travel,
                        flap_x, flap_y, rod_length, grid_points=grid_points)
    t0, t1 = res["flap_angle_rad"]
    if np.isnan(t0) or np.isnan(t1):
        return float("inf")
    return float(t0 + t1)


def auto_rod_length(servo_x: float, servo_y: float, servo_travel: float,
                    flap_x: float, flap_y: float,
                    samples: int = 50,
                    grid_points: int = 200) -> float:
    """Compute rod length so θ(0) ≈ -θ(1) (symmetric flap range about 0°)."""
    r = np.hypot(flap_x, flap_y)
    u_grid = np.linspace(0, 1, 50)
    d_grid = np.hypot(servo_x + u_grid * servo_travel, servo_y)

    L_low = float(np.max(np.abs(d_grid - r)))
    L_high = float(np.min(d_grid + r))
    if not (L_low < L_high):
        return (L_low + L_high) * 0.5

    Ls = np.linspace(L_low, L_high, samples)
    f = np.full(samples, np.nan)
    for i, L in enumerate(Ls):
        f[i] = _endpoint_angle_sum(L, servo_x, servo_y, servo_travel,
                                   flap_x, flap_y, grid_points=grid_points)

    # Find a sign-change bracket and refine with brentq.
    best_abs = float("inf")
    best_idx = None
    for i in range(samples - 1):
        if np.isnan(f[i]) or np.isnan(f[i + 1]):
            continue
        if abs(f[i]) < best_abs:
            best_abs = abs(f[i])
            best_idx = i
        if abs(f[i + 1]) < best_abs:
            best_abs = abs(f[i + 1])
            best_idx = i + 1
        if f[i] * f[i + 1] <= 0.0:
            try:
                return brentq(
                    lambda L: _endpoint_angle_sum(L, servo_x, servo_y, servo_travel,
                                                  flap_x, flap_y, grid_points=grid_points),
                    float(Ls[i]),
                    float(Ls[i + 1]),
                    xtol=1e-6,
                )
            except ValueError:
                pass

    if best_idx is not None:
        return float(Ls[best_idx])
    return (L_low + L_high) * 0.5


def simulate_flap(inputs: FloatArray,
                  servo_x: float,
                  servo_y: float,
                  servo_travel: float,
                  flap_x: float,
                  flap_y: float,
                  rod_length: float,
                  initial_angle: float = 0.0,
                  angle_bounds=(-np.pi, np.pi),
                  grid_points: int = 600) -> dict:
    """Compute flap angle and geometry for given servo inputs.

    Parameters
    ----------
    inputs : float or array_like
        Normalized servo command(s) in [0, 1].
    servo_x : float
        Servo endpoint x-position at input 0.
    servo_y : float
        Servo endpoint y-position (constant).
    servo_travel : float
        Distance the servo endpoint moves in +x from input 0 to input 1.
    flap_x, flap_y : float
        Control-rod attachment offsets in the flap's local frame.
    rod_length : float
        Length of the control rod.
    initial_angle : float, optional
        Initial guess / starting branch for the flap angle (radians).

    Returns
    -------
    dict
        Arrays/values for servo_input, servo_x, servo_y, flap_angle_rad,
        flap_angle_deg, attach_x, attach_y, rod_length_check, and valid mask.
    """
    u = np.asarray(inputs, dtype=float)
    scalar_input = u.ndim == 0
    if scalar_input:
        u = u.reshape(1)

    sorted_idx = np.argsort(u)
    u_sorted = u[sorted_idx]

    theta = _solve_theta(
        u_sorted, servo_x, servo_y, servo_travel,
        flap_x, flap_y, rod_length, initial_angle,
        angle_bounds=angle_bounds, grid_points=grid_points,
    )

    Sx_sorted = servo_x + u_sorted * servo_travel
    Sy_sorted = np.full_like(u_sorted, servo_y)

    attach_x = np.full_like(u_sorted, np.nan)
    attach_y = np.full_like(u_sorted, np.nan)
    valid = ~np.isnan(theta)
    for i in np.where(valid)[0]:
        attach_x[i], attach_y[i] = _attach_point(float(theta[i]), flap_x, flap_y)

    rod_check = np.hypot(Sx_sorted - attach_x, Sy_sorted - attach_y)

    inv_idx = np.empty_like(sorted_idx)
    inv_idx[sorted_idx] = np.arange(sorted_idx.size)

    def restore(arr):
        return arr[inv_idx]

    result = {
        "servo_input": restore(u_sorted),
        "servo_x": restore(Sx_sorted),
        "servo_y": restore(Sy_sorted),
        "flap_angle_rad": restore(theta),
        "flap_angle_deg": np.degrees(restore(theta)),
        "attach_x": restore(attach_x),
        "attach_y": restore(attach_y),
        "rod_length_check": restore(rod_check),
        "valid": restore(valid),
    }

    if scalar_input:
        result = {k: (v.item() if isinstance(v, np.ndarray) else v)
                  for k, v in result.items()}
    return result
