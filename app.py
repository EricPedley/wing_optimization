"""Interactive Plotly Dash app for the flap-servo simulation.

The geometry solve and the optimizer both run through the JAX model in
:mod:`fastmodel`, which is fast enough (~0.2 s for a full 128-start search) that
optimization runs automatically whenever an input settles, rather than sitting
behind a button.

The design sliders are the *baseline* geometry and the anchors that "<=" and
">=" constraints are measured against, so the optimizer never writes back to
them; its result is held separately and applied only on request.
"""

import threading
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update

import fastmodel as fm
import fastopt as fo

PARAMS = [
    {"id": "servo-x", "name": "Servo start x", "min": 10, "max": 20, "step": 0.1, "value": 15.0},
    {"id": "servo-y", "name": "Servo y", "min": 0, "max": 10, "step": 0.1, "value": 6.0},
    {"id": "servo-travel", "name": "Servo travel x", "min": 5, "max": 15, "step": 0.1, "value": 9.0},
    {"id": "flap-x", "name": "Flap attach x", "min": -5, "max": 20, "step": 0.1, "value": 0.0},
    {"id": "flap-y", "name": "Flap attach y", "min": 5, "max": 10, "step": 0.1, "value": 10.0},
    # Live cursor: redraws while dragging rather than on release.
    {"id": "current-input", "name": "Servo input", "min": 0, "max": 1, "step": 0.01,
     "value": 0.0, "design": False, "updatemode": "drag"},
]

# Sliders the optimizer is allowed to move.  "current-input" is only a viewing
# cursor, so it gets no constraint dropdown and sits under the layout plot.
DESIGN_PARAMS = [p for p in PARAMS if p.get("design", True)]
INPUT_PARAM = next(p for p in PARAMS if not p.get("design", True))

CONSTRAINT_OPTIONS = [
    {"label": "= (locked)", "value": "fixed"},
    {"label": "≤ slider", "value": "le"},
    {"label": "≥ slider", "value": "ge"},
    {"label": "unconstrained", "value": "free"},
]

OBJECTIVE_OPTIONS = [
    {"label": "No optimization", "value": "none"},
    {"label": "Area under mechanical-advantage curve", "value": "area"},
    {"label": "Peak mechanical advantage", "value": "peak"},
    {"label": "Minimum mechanical advantage", "value": "min"},
]

OBJECTIVE_LABELS = {"area": "Area", "peak": "Peak", "min": "Minimum"}

SOFT_OPTIONS = [
    {"label": "Peak advantage occurs at flap angle = 0", "value": "peak_at_zero"},
    {"label": "Advantage at min angle == advantage at max angle", "value": "symmetric_ends"},
]

N_CURVE = 201


def _var_name(param_id):
    """Slider id ('servo-x') to optimizer variable name ('servo_x')."""
    return param_id.replace("-", "_")


def _slider(p):
    row = [
        dcc.Slider(
            id=p["id"],
            min=p["min"],
            max=p["max"],
            step=p["step"],
            value=p["value"],
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode=p.get("updatemode", "mouseup"),
        )
    ]
    children = [html.Label(p["name"], style={"fontWeight": "bold"})]

    if p.get("design", True):
        children.append(html.Div(
            [
                html.Div(row, style={"flex": "1 1 auto", "minWidth": "0"}),
                dcc.Dropdown(
                    id=f"{p['id']}-constraint",
                    options=CONSTRAINT_OPTIONS,
                    value="fixed",
                    clearable=False,
                    style={"width": "150px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "10px"},
        ))
    else:
        children.extend(row)

    return html.Div(children, style={"padding": "10px"})


app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Airplane Flap Dynamics"),
        html.Div(
            [_slider(p) for p in DESIGN_PARAMS],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Objective", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="objective",
                            options=OBJECTIVE_OPTIONS,
                            value="none",
                            clearable=False,
                        ),
                        html.Button(
                            "Apply result to sliders",
                            id="apply-result",
                            n_clicks=0,
                            style={"marginTop": "10px", "padding": "8px 14px",
                                   "fontSize": "0.95em", "width": "100%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Nice-to-have constraints", style={"fontWeight": "bold"}),
                        dcc.Checklist(
                            id="soft-constraints",
                            options=SOFT_OPTIONS,
                            value=[],
                            labelStyle={"display": "block"},
                        ),
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="min-angle-enable",
                                    options=[{"label": "Min. peak flap angle (°)",
                                              "value": "on"}],
                                    value=[],
                                    style={"whiteSpace": "nowrap"},
                                ),
                                html.Div(
                                    dcc.Slider(
                                        id="min-angle",
                                        min=10,
                                        max=45,
                                        step=1,
                                        value=20,
                                        marks={10: "10", 45: "45"},
                                        tooltip={"placement": "bottom",
                                                 "always_visible": False},
                                    ),
                                    id="min-angle-wrap",
                                    style={"flex": "1 1 auto", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "10px", "paddingTop": "5px"},
                        ),
                    ]
                ),
                dcc.Loading(
                    html.Div(id="optimize-status"),
                    type="dot",
                    delay_show=150,
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                   "gap": "20px", "padding": "10px",
                   "border": "1px solid #ddd", "borderRadius": "6px",
                   "margin": "10px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="physical-graph", style={"height": "550px"}),
                        _slider(INPUT_PARAM),
                    ]
                ),
                dcc.Graph(id="curve-graph", style={"height": "550px"}),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
        ),
        html.Div(id="angle-display", style={"padding": "10px", "fontSize": "1.2em"}),
        # Geometry actually being plotted: the sliders, or the optimizer's answer.
        dcc.Store(id="geometry"),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto"},
)


@callback(
    Output("min-angle", "disabled"),
    Output("min-angle-wrap", "className"),
    Input("min-angle-enable", "value"),
)
def toggle_min_angle(enabled):
    on = bool(enabled)
    return not on, "" if on else "slider-off"


@callback(
    Output("geometry", "data"),
    Output("optimize-status", "children"),
    [Input(p["id"], "value") for p in DESIGN_PARAMS],
    [Input(f"{p['id']}-constraint", "value") for p in DESIGN_PARAMS],
    Input("objective", "value"),
    Input("soft-constraints", "value"),
    Input("min-angle-enable", "value"),
    Input("min-angle", "value"),
)
def run_optimization(*state):
    n = len(DESIGN_PARAMS)
    slider_values = state[:n]
    constraint_modes = state[n:2 * n]
    objective, soft = state[2 * n], state[2 * n + 1] or []
    angle_enabled, min_angle = state[2 * n + 2], state[2 * n + 3]

    values = {_var_name(p["id"]): float(v) for p, v in zip(DESIGN_PARAMS, slider_values)}
    modes = {_var_name(p["id"]): m for p, m in zip(DESIGN_PARAMS, constraint_modes)}
    ranges = {_var_name(p["id"]): (float(p["min"]), float(p["max"])) for p in DESIGN_PARAMS}
    min_max_angle = float(min_angle) if angle_enabled else None

    if objective == "none":
        return ({"values": values, "optimized": False},
                html.Div("Optimization off — showing the slider geometry.",
                         style={"color": "#666"}))

    result = fo.optimize(values, modes, ranges, objective, soft,
                         min_max_angle=min_max_angle)
    return ({"values": result["values"], "optimized": True},
            _optimize_status(result, objective, min_max_angle))


@callback(
    [Output(p["id"], "value") for p in DESIGN_PARAMS],
    Input("apply-result", "n_clicks"),
    State("geometry", "data"),
    prevent_initial_call=True,
)
def apply_result(_n_clicks, geometry):
    """Copy the optimized geometry onto the sliders, on explicit request.

    This is deliberately manual: the sliders anchor the '<=' and '>=' bounds, so
    writing to them automatically would drag the constraints along with each run.
    """
    if not geometry or not geometry.get("optimized"):
        return [no_update] * len(DESIGN_PARAMS)
    values = geometry["values"]
    return [round(values[_var_name(p["id"])], 3) for p in DESIGN_PARAMS]


def _optimize_status(result, objective, min_max_angle=None):
    start, best = result["start_metrics"], result["best_metrics"]
    lines = [
        html.Div(f"{result['message']}  ({result['elapsed'] * 1000:.0f} ms, "
                 f"{result['n_free']} free variable(s))",
                 style={"fontWeight": "bold"}),
    ]

    if objective in OBJECTIVE_LABELS:
        label = OBJECTIVE_LABELS[objective]
        lines.append(html.Div(f"{label}: {start[objective]:.3f} → {best[objective]:.3f}"))
    lines.append(html.Div(
        f"Peak at flap angle: {np.degrees(start['theta_at_peak']):.2f}° → "
        f"{np.degrees(best['theta_at_peak']):.2f}°"
    ))
    lines.append(html.Div(
        f"Advantage at min/max angle: {start['mag_at_min_angle']:.2f}/"
        f"{start['mag_at_max_angle']:.2f} → {best['mag_at_min_angle']:.2f}/"
        f"{best['mag_at_max_angle']:.2f}"
    ))

    text = (f"Peak flap angle: {start['max_angle_deg']:.2f}° → "
            f"{best['max_angle_deg']:.2f}°")
    style = None
    if min_max_angle:
        met = best["max_angle_deg"] >= min_max_angle - 0.05
        text += f"  (need ≥ {min_max_angle:.0f}° — {'met' if met else 'NOT met'})"
        style = {"color": "green" if met else "crimson"}
    lines.append(html.Div(text, style=style))

    if best["valid_fraction"] < 1.0:
        lines.append(html.Div(
            f"Warning: linkage cannot close over {(1 - best['valid_fraction']) * 100:.0f}%"
            " of the servo stroke.", style={"color": "crimson"}))

    lines.append(html.Div(
        "  ".join(f"{p['name']}={best_v:.3f}"
                  for p, best_v in zip(DESIGN_PARAMS,
                                       (result["values"][_var_name(p["id"])]
                                        for p in DESIGN_PARAMS))),
        style={"paddingTop": "6px", "fontFamily": "monospace", "fontSize": "0.85em"},
    ))
    return lines


def _flap_len(flap_x, flap_y):
    return max(np.hypot(flap_x, flap_y) * 1.5, 2.0)


def _padded(lo, hi, pad=0.05):
    """An axis range covering [lo, hi] with a margin, safe on degenerate spans."""
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    span = hi - lo
    margin = pad * span if span > 1e-9 else max(abs(hi), 1.0) * pad
    return [lo - margin, hi + margin]


def _plot_ranges(theta, ratio, attach_x, attach_y, ok,
                 servo_x, servo_y, servo_travel, flap_x, flap_y):
    """Axis ranges covering everything either plot can draw over the full sweep.

    Fixing these to the sweep keeps the axes still while the input cursor moves,
    since only the cursor-dependent traces would otherwise resize them.
    """
    th = theta[ok]
    r_attach = np.hypot(flap_x, flap_y)
    flap_len = _flap_len(flap_x, flap_y)

    xs = [0.0, servo_x, servo_x + servo_travel, -r_attach, r_attach, -flap_len]
    ys = [0.0, servo_y, -r_attach, r_attach, 0.0]
    if th.size:
        xs.extend(-flap_len * np.cos(th))
        ys.extend(flap_len * np.sin(th))
        xs.extend(attach_x[ok])
        ys.extend(attach_y[ok])

    angles = np.degrees(th)
    finite = ratio[ok][np.isfinite(ratio[ok])] if th.size else np.array([])

    return {
        "physical_x": _padded(np.min(xs), np.max(xs)),
        "physical_y": _padded(np.min(ys), np.max(ys)),
        "angle_y": _padded(np.min(angles), np.max(angles)) if angles.size else None,
        "ratio_y": _padded(np.min(finite), np.max(finite)) if finite.size else None,
    }


@lru_cache(maxsize=64)
def _sweep(servo_x, servo_y, servo_travel, flap_x, flap_y):
    """Everything the plots need for one geometry.

    The "Servo input" slider redraws while being dragged and does not change the
    geometry, so this is cached and only the cursor index moves.
    """
    p = np.array([servo_x, servo_y, servo_travel, flap_x, flap_y], dtype=np.float64)
    rod_length, u, theta, ratio, ax, ay, ok = fm.display_sweep(p, N_CURVE)
    out = (float(rod_length), np.asarray(u), np.asarray(theta), np.asarray(ratio),
           np.asarray(ax), np.asarray(ay), np.asarray(ok))
    ranges = _plot_ranges(out[2], out[3], out[4], out[5], out[6],
                          servo_x, servo_y, servo_travel, flap_x, flap_y)
    return out, ranges


@callback(
    Output("physical-graph", "figure"),
    Output("curve-graph", "figure"),
    Output("angle-display", "children"),
    Input("geometry", "data"),
    Input("current-input", "value"),
)
def update(geometry, current_input):
    if not geometry:
        return no_update, no_update, no_update
    v = geometry["values"]
    servo_x, servo_y, servo_travel, flap_x, flap_y = (
        v["servo_x"], v["servo_y"], v["servo_travel"], v["flap_x"], v["flap_y"])

    (rod_length, u, theta, ratio, attach_x, attach_y, ok), ranges = _sweep(
        servo_x, servo_y, servo_travel, flap_x, flap_y)

    current_idx = int(np.argmin(np.abs(u - current_input)))
    valid = bool(ok[current_idx])

    physical = _build_physical_figure(
        valid, float(theta[current_idx]), attach_x[current_idx], attach_y[current_idx],
        servo_x, servo_y, servo_travel, current_input, flap_x, flap_y, ranges,
    )
    curve = _build_curve_figure(u, np.degrees(theta), ratio, ok, current_idx, ranges)

    source = "optimized" if geometry.get("optimized") else "sliders"
    if valid:
        angle_text = (
            f"Servo input = {current_input:.2f}  |  "
            f"Flap angle = {np.degrees(theta[current_idx]):.2f}°  |  "
            f"Rod length = {rod_length:.3f}  |  showing {source} geometry"
        )
    else:
        angle_text = (f"Servo input = {current_input:.2f}  |  No valid geometry  |  "
                      f"Rod length = {rod_length:.3f}  |  showing {source} geometry")

    return physical, curve, angle_text


def _build_physical_figure(valid, theta, attach_x, attach_y,
                           servo_x, servo_y, servo_travel, current_input,
                           flap_x, flap_y, ranges):
    fig = go.Figure()

    flap_len = _flap_len(flap_x, flap_y)
    if valid:
        flap_end_x = -flap_len * np.cos(theta)
        flap_end_y = flap_len * np.sin(theta)
    else:
        flap_end_x = -flap_len
        flap_end_y = 0.0

    # Flap chord
    fig.add_trace(
        go.Scatter(
            x=[0.0, flap_end_x],
            y=[0.0, flap_end_y],
            mode="lines",
            line={"color": "black", "width": 4},
            name="Flap",
            hoverinfo="skip",
        )
    )

    # Hinge
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers",
            marker={"color": "black", "size": 10},
            name="Hinge",
        )
    )

    # Attachment circle (locus of the rod pin as the flap rotates)
    r_attach = np.hypot(flap_x, flap_y)
    t_circle = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=r_attach * np.cos(t_circle),
            y=r_attach * np.sin(t_circle),
            mode="lines",
            line={"color": "gray", "width": 1, "dash": "dash"},
            name="Attach locus",
            hoverinfo="skip",
        )
    )

    if valid:
        # Rod
        sx_cur = servo_x + current_input * servo_travel
        sy_cur = servo_y
        fig.add_trace(
            go.Scatter(
                x=[sx_cur, attach_x],
                y=[sy_cur, attach_y],
                mode="lines",
                line={"color": "blue", "width": 3},
                name="Control rod",
                hoverinfo="skip",
            )
        )
        # Rod ends
        fig.add_trace(
            go.Scatter(
                x=[attach_x],
                y=[attach_y],
                mode="markers",
                marker={"color": "blue", "size": 10},
                name="Rod pin",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[sx_cur],
                y=[sy_cur],
                mode="markers",
                marker={"color": "green", "size": 10},
                name="Servo endpoint",
            )
        )

    # Servo rail
    rail_x0 = servo_x
    rail_x1 = servo_x + servo_travel
    fig.add_trace(
        go.Scatter(
            x=[rail_x0, rail_x1],
            y=[servo_y, servo_y],
            mode="lines",
            line={"color": "red", "width": 2, "dash": "dash"},
            name="Servo rail",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[rail_x0, rail_x1],
            y=[servo_y, servo_y],
            mode="markers",
            marker={"color": "red", "size": 8},
            name="Servo limits",
        )
    )

    fig.update_layout(
        title="Physical layout",
        # Ranges come from the whole sweep so the view holds still as the input
        # cursor moves.  scaleanchor keeps the aspect square; Plotly may widen
        # one axis past the request to honour it, which is fine and stable.
        xaxis={"title": "x", "range": ranges["physical_x"], "autorange": False},
        yaxis={"title": "y", "scaleanchor": "x", "scaleratio": 1,
               "range": ranges["physical_y"], "autorange": False},
        showlegend=True,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        uirevision="physical",
    )
    return fig


def _build_curve_figure(u, angle_deg, ratio, ok, current_idx, ranges):
    y = np.where(ok, angle_deg, np.nan)
    r = np.where(ok, ratio, np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u,
            y=y,
            mode="lines",
            line={"color": "blue", "width": 2},
            name="Flap angle",
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u,
            y=r,
            mode="lines",
            line={"color": "orange", "width": 2},
            name="Torque / servo force",
            yaxis="y2",
            connectgaps=False,
        )
    )
    if ok[current_idx]:
        fig.add_trace(
            go.Scatter(
                x=[u[current_idx]],
                y=[y[current_idx]],
                mode="markers",
                marker={"color": "red", "size": 12},
                name="Current input",
            )
        )
        if np.isfinite(r[current_idx]):
            fig.add_trace(
                go.Scatter(
                    x=[u[current_idx]],
                    y=[r[current_idx]],
                    mode="markers",
                    marker={"color": "darkorange", "size": 12},
                    name="Current ratio",
                    yaxis="y2",
                )
            )

    fig.update_layout(
        title="Servo input vs. flap angle and mechanical advantage",
        xaxis={"title": "Servo input (0 → 1)", "range": [0, 1]},
        yaxis={"title": {"text": "Flap angle (degrees)", "font": {"color": "blue"}},
               "tickfont": {"color": "blue"},
               "range": ranges["angle_y"], "autorange": ranges["angle_y"] is None},
        yaxis2={
            "title": {"text": "Torque / servo force (length)", "font": {"color": "orange"}},
            "tickfont": {"color": "orange"},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": ranges["ratio_y"],
            "autorange": ranges["ratio_y"] is None,
        },
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 40, "r": 60, "t": 60, "b": 40},
        uirevision="curve",
    )
    return fig


# XLA compilation of the solver takes ~18 s.  Doing it in the background lets the
# page load immediately; the first optimization simply waits for it to finish.
threading.Thread(target=fo.warmup, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True, port=8050)
