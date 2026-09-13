import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _finite_array(value, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _time(t):
    t = _finite_array(t, "t")
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must be one-dimensional with at least two strictly increasing samples")
    if not np.isfinite(t[-1] - t[0]):
        raise ValueError("t duration must be finite")
    return t


def _samples(value, name, lengths):
    array = _finite_array(value, name)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] not in lengths or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (samples,) or (samples, axes), with samples in {lengths}")
    return array


def _tracking_inputs(t, reference, rates):
    t = _time(t)
    reference = _samples(reference, "reference", (len(t),))
    rates = _samples(rates, "rates", (len(t),))
    if reference.shape != rates.shape:
        raise ValueError("reference and rates must have matching shapes")
    return t, reference, rates


def _vector(value, name, lengths):
    array = _finite_array(value, name)
    if array.ndim != 1 or len(array) not in lengths:
        raise ValueError(f"{name} must be one-dimensional with length in {lengths}")
    return array


def _bounds(value, name, axes):
    array = _finite_array(value, name)
    if array.ndim > 1 or (array.ndim == 1 and len(array) not in (1, axes)):
        raise ValueError(f"{name} must be scalar or have one value per command axis")
    return np.broadcast_to(array, (axes,))


def tracking_metrics(t, reference, rates, commands, command_lower, command_upper,
                     solve_times_s=None, solver_success=None, dt=None):
    t, reference, rates = _tracking_inputs(t, reference, rates)
    commands = _samples(commands, "commands", (len(t) - 1, len(t)))
    lower = _bounds(command_lower, "command_lower", commands.shape[1])
    upper = _bounds(command_upper, "command_upper", commands.shape[1])
    if np.any(lower >= upper):
        raise ValueError("command_lower must be strictly less than command_upper")
    if dt is not None:
        dt_array = _finite_array(dt, "dt")
        if dt_array.ndim != 0 or dt_array <= 0:
            raise ValueError("dt must be a positive finite scalar")
        dt = float(dt_array)
    intervals = np.diff(t)
    duration = float(t[-1] - t[0])
    error = np.rad2deg(rates - reference)
    held_commands = commands[:len(t) - 1]
    saturated = np.any((held_commands <= lower) | (held_commands >= upper), axis=1)
    saturation_duration = float(np.sum(intervals * saturated))
    result = {
        "rms_error_deg_s": float(np.sqrt(np.trapezoid(np.mean(error ** 2, axis=1), t) / duration)),
        "iae_deg": float(np.trapezoid(np.sum(np.abs(error), axis=1), t)),
        "peak_error_deg_s": float(np.max(np.abs(error))),
        "command_effort": float(np.sum(intervals * np.sum(held_commands ** 2, axis=1))),
        "command_total_variation": float(np.sum(np.abs(np.diff(commands, axis=0)))),
        "saturation_fraction": saturation_duration / duration,
        "saturation_duration_s": saturation_duration,
        "max_solver_wall_ms": None,
        "p95_solver_wall_ms": None,
        "deadline_miss_fraction": None,
        "solver_failure_count": None,
    }
    solve_times = None
    if solve_times_s is not None:
        solve_times = _vector(solve_times_s, "solve_times_s", (len(t) - 1, len(t)))
        if np.any(solve_times < 0):
            raise ValueError("solve_times_s must be nonnegative")
        deadlines = dt if dt is not None else np.r_[intervals, intervals[-1]][:len(solve_times)]
        result.update({
            "max_solver_wall_ms": float(np.max(solve_times) * 1000),
            "p95_solver_wall_ms": float(np.percentile(solve_times, 95) * 1000),
            "deadline_miss_fraction": float(np.mean(solve_times > deadlines)),
        })
    if solver_success is not None:
        lengths = (len(solve_times),) if solve_times is not None else (len(t) - 1, len(t))
        success = _vector(solver_success, "solver_success", lengths)
        if np.any((success != 0) & (success != 1)):
            raise ValueError("solver_success must contain only booleans or 0/1")
        result["solver_failure_count"] = int(np.count_nonzero(success == 0))
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError("inputs produce nonfinite metrics")
    return result


def _crossing(t, progress, threshold):
    attained = np.flatnonzero(progress >= threshold)
    if not len(attained):
        return None
    index = int(attained[0])
    if index == 0:
        return float(t[0])
    fraction = (threshold - progress[index - 1]) / (progress[index] - progress[index - 1])
    return float(t[index - 1] + fraction * (t[index] - t[index - 1]))


def _single_step(t, reference, rates):
    result = dict(step_onset_s=None, rise_time_s=None, overshoot_percent=None, settling_time_s=None)
    tolerance = 1e-10 * max(float(np.max(np.abs(reference))), 1e-12)
    changes = np.flatnonzero(np.abs(np.diff(reference)) > tolerance) + 1
    if len(changes) != 1:
        return result
    onset = int(changes[0])
    if len(t) - onset < 2:
        return result
    initial, target = reference[0], reference[-1]
    amplitude = target - initial
    if not np.isfinite(amplitude):
        raise ValueError("step amplitude must be finite")
    if abs(amplitude) <= tolerance:
        return result
    if (np.any(np.abs(reference[:onset] - initial) > tolerance)
            or np.any(np.abs(reference[onset:] - target) > tolerance)):
        return result
    elapsed = t[onset:] - t[onset]
    progress = (rates[onset:] - initial) / amplitude
    if not np.all(np.isfinite(progress)):
        raise ValueError("inputs produce nonfinite step response")
    low, high = _crossing(elapsed, progress, 0.1), _crossing(elapsed, progress, 0.9)
    result["step_onset_s"] = float(t[onset] - t[0])
    if low is not None and high is not None:
        result["rise_time_s"] = high - low
    result["overshoot_percent"] = float(max(0.0, np.max(progress) - 1.0) * 100)
    outside = np.flatnonzero(np.abs(progress - 1.0) > 0.02 + 1e-12)
    first_settled = int(outside[-1] + 1) if len(outside) else 0
    if first_settled < len(elapsed) - 1:
        if first_settled == 0:
            result["settling_time_s"] = 0.0
        else:
            previous = first_settled - 1
            boundary = 0.98 if progress[previous] < 0.98 else 1.02
            fraction = (boundary - progress[previous]) / (progress[first_settled] - progress[previous])
            result["settling_time_s"] = float(
                elapsed[previous] + fraction * (elapsed[first_settled] - elapsed[previous]))
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError("inputs produce nonfinite step metrics")
    return result


def step_metrics(t, reference, rates):
    t, reference, rates = _tracking_inputs(t, reference, rates)
    per_axis = [_single_step(t, reference[:, axis], rates[:, axis])
                for axis in range(reference.shape[1])]
    if len(per_axis) == 1:
        return per_axis[0]
    return {key: [axis[key] for axis in per_axis] for key in per_axis[0]}


def _metadata(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("report rows must contain flat, finite JSON scalar values")


def write_report(rows, traces, output_path, title="Optimal Rate Control"):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    rows = [{str(key): _metadata(value) for key, value in row.items()} for row in rows]
    prepared = []
    for trace in traces:
        t, reference, rates = _tracking_inputs(trace["t"], trace["reference"], trace["rates"])
        commands = _samples(trace["commands"], "commands", (len(t) - 1, len(t)))
        timing = trace.get("solve_times_s")
        if timing is not None:
            timing = _vector(timing, "solve_times_s", (len(t) - 1, len(t)))
            if np.any(timing < 0):
                raise ValueError("solve_times_s must be nonnegative")
        prepared.append((html.escape(str(trace["label"])), t, reference, rates, commands, timing))
    has_timing = any(trace[-1] is not None for trace in prepared)
    subplot_titles = ["Tracking versus command effort", "Angular rate tracking", "Commands"]
    if has_timing:
        subplot_titles.append("Solver wall time")
    figure = make_subplots(rows=len(subplot_titles), cols=1, subplot_titles=subplot_titles,
                           vertical_spacing=0.08)
    for index, row in enumerate(rows):
        if "command_effort" not in row or "rms_error_deg_s" not in row:
            raise ValueError("each report row requires command_effort and rms_error_deg_s")
        for key in ("command_effort", "rms_error_deg_s"):
            if not isinstance(row[key], (int, float)) or isinstance(row[key], bool) or row[key] < 0:
                raise ValueError(f"{key} must be a nonnegative finite number")
        label = html.escape(str(row.get("label", row.get("controller", f"Run {index + 1}"))))
        details = "<br>".join(f"{html.escape(key)}: {html.escape(str(value))}" for key, value in row.items())
        figure.add_trace(go.Scatter(
            x=[row["command_effort"]], y=[row["rms_error_deg_s"]], mode="markers", name=label,
            text=[details], hovertemplate="%{text}<extra></extra>", marker={"size": 10}), row=1, col=1)
    for index, (label, t, reference, rates, commands, timing) in enumerate(prepared):
        group = f"trace-{index}"
        for axis in range(rates.shape[1]):
            suffix = f"axis {axis + 1}"
            figure.add_trace(go.Scatter(
                x=t, y=np.rad2deg(reference[:, axis]), mode="lines", name=f"{label} reference {suffix}",
                legendgroup=group, line={"dash": "dash"}), row=2, col=1)
            figure.add_trace(go.Scatter(
                x=t, y=np.rad2deg(rates[:, axis]), mode="lines", name=f"{label} rate {suffix}",
                legendgroup=group), row=2, col=1)
        for axis in range(commands.shape[1]):
            command_t = t
            values = commands[:, axis]
            if len(commands) == len(t) - 1:
                values = np.r_[values, values[-1]]
            figure.add_trace(go.Scatter(
                x=command_t, y=values, mode="lines", name=f"{label} command {axis + 1}",
                legendgroup=group, line={"shape": "hv"}), row=3, col=1)
        if timing is not None:
            figure.add_trace(go.Scatter(
                x=t[:len(timing)], y=timing * 1000, mode="lines", name=f"{label} solve time",
                legendgroup=group), row=4, col=1)
    figure.update_xaxes(title_text="Command effort (fraction² s, summed over commands)", row=1, col=1)
    figure.update_yaxes(title_text="RMS error (deg/s)", row=1, col=1)
    figure.update_yaxes(title_text="Rate (deg/s)", row=2, col=1)
    figure.update_yaxes(title_text="Command fraction", row=3, col=1)
    for row_index in range(2, len(subplot_titles) + 1):
        figure.update_xaxes(title_text="Time (s)", row=row_index, col=1,
                            matches="x2" if row_index > 2 else None)
    if has_timing:
        figure.update_yaxes(title_text="Wall time (ms)", row=4, col=1)
    safe_title = html.escape(str(title))
    figure.update_layout(title=safe_title, height=350 * len(subplot_titles), template="plotly_white",
                         hovermode="closest", legend={"groupclick": "toggleitem"})
    metadata = html.escape(json.dumps(rows, indent=2, allow_nan=False))
    plot = figure.to_html(full_html=False, include_plotlyjs=True)
    document = (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{safe_title}</title>'
                f'</head><body><h1>{safe_title}</h1>{plot}'
                f'<details><summary>Metrics and configuration</summary><pre>{metadata}</pre></details>'
                '</body></html>')
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return output_path
