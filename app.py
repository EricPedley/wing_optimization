"""Interactive Plotly Dash app for the flap-servo simulation."""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html

import optimize as opt
from core import auto_rod_length, simulate_flap, torque_force_ratio

PARAMS = [
    {"id": "servo-x", "name": "Servo start x", "min": 10, "max": 20, "step": 0.1, "value": 15.0},
    {"id": "servo-y", "name": "Servo y", "min": 0, "max": 10, "step": 0.1, "value": 6.0},
    {"id": "servo-travel", "name": "Servo travel x", "min": 5, "max": 15, "step": 0.1, "value": 9.0},
    {"id": "flap-x", "name": "Flap attach x", "min": -5, "max": 20, "step": 0.1, "value": 0.0},
    {"id": "flap-y", "name": "Flap attach y", "min": 5, "max": 10, "step": 0.1, "value": 10.0},
    {"id": "current-input", "name": "Servo input", "min": 0, "max": 1, "step": 0.01,
     "value": 0.0, "design": False},
]

# Sliders the optimizer is allowed to move.  "current-input" is only a viewing
# cursor, so it gets no constraint dropdown.
DESIGN_PARAMS = [p for p in PARAMS if p.get("design", True)]

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
        )
    ]
    children = [html.Label(p["name"], style={"fontWeight": "bold"})]

    if p.get("design", True):
        body = html.Div(
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
        )
        children.append(body)
    else:
        children.extend(row)

    return html.Div(children, style={"padding": "10px"})


app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Airplane Flap Dynamics"),
        html.Div(
            [_slider(p) for p in PARAMS],
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
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            "Run optimization",
                            id="run-optimize",
                            n_clicks=0,
                            style={"padding": "10px 20px", "fontSize": "1em"},
                        ),
                        dcc.Loading(
                            html.Div(id="optimize-status", style={"paddingTop": "10px"}),
                            type="dot",
                        ),
                    ]
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                   "gap": "20px", "padding": "10px",
                   "border": "1px solid #ddd", "borderRadius": "6px",
                   "margin": "10px"},
        ),
        html.Div(
            [
                dcc.Graph(id="physical-graph", style={"height": "550px"}),
                dcc.Graph(id="curve-graph", style={"height": "550px"}),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
        ),
        html.Div(id="angle-display", style={"padding": "10px", "fontSize": "1.2em"}),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto"},
)


@callback(
    Output("run-optimize", "disabled"),
    Input("objective", "value"),
)
def toggle_run_button(objective):
    return objective == "none"


@callback(
    [Output(p["id"], "value") for p in DESIGN_PARAMS],
    Output("optimize-status", "children"),
    Input("run-optimize", "n_clicks"),
    [State(p["id"], "value") for p in DESIGN_PARAMS],
    [State(f"{p['id']}-constraint", "value") for p in DESIGN_PARAMS],
    State("objective", "value"),
    State("soft-constraints", "value"),
    running=[(Output("run-optimize", "disabled"), True, False)],
    prevent_initial_call=True,
)
def run_optimization(_n_clicks, *state):
    n = len(DESIGN_PARAMS)
    slider_values = state[:n]
    constraint_modes = state[n:2 * n]
    objective, soft = state[2 * n], state[2 * n + 1] or []

    values = {_var_name(p["id"]): v for p, v in zip(DESIGN_PARAMS, slider_values)}
    modes = {_var_name(p["id"]): m for p, m in zip(DESIGN_PARAMS, constraint_modes)}
    ranges = {_var_name(p["id"]): (p["min"], p["max"]) for p in DESIGN_PARAMS}

    result = opt.optimize(values, modes, ranges, objective, soft)
    new_values = [round(result["values"][_var_name(p["id"])], 3) for p in DESIGN_PARAMS]

    return [*new_values, _optimize_status(result, objective)]


def _optimize_status(result, objective):
    start, best = result["start_metrics"], result["best_metrics"]
    lines = [
        html.Div(f"{result['message']}  ({result['elapsed']:.1f}s, "
                 f"{result['n_free']} free variable(s))"),
    ]
    if start is None or best is None:
        lines.append(html.Div("No valid geometry to score.", style={"color": "crimson"}))
        return lines

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
    return lines


@callback(
    Output("physical-graph", "figure"),
    Output("curve-graph", "figure"),
    Output("angle-display", "children"),
    [Input(p["id"], "value") for p in PARAMS],
)
def update(servo_x, servo_y, servo_travel, flap_x, flap_y, current_input):
    rod_length = auto_rod_length(servo_x, servo_y, servo_travel, flap_x, flap_y)

    curve_inputs = np.linspace(0, 1, 201)
    all_inputs = np.sort(np.unique(np.concatenate([curve_inputs, [current_input]])))

    res = simulate_flap(
        all_inputs,
        servo_x=servo_x,
        servo_y=servo_y,
        servo_travel=servo_travel,
        flap_x=flap_x,
        flap_y=flap_y,
        rod_length=rod_length,
    )

    current_idx = int(np.argmin(np.abs(all_inputs - current_input)))
    theta = float(res["flap_angle_rad"][current_idx])
    valid = bool(res["valid"][current_idx])

    physical = _build_physical_figure(
        valid, theta, res["attach_x"][current_idx], res["attach_y"][current_idx],
        servo_x, servo_y, servo_travel, current_input, flap_x, flap_y,
    )
    curve = _build_curve_figure(res, current_idx, servo_travel)

    if valid:
        angle_text = (
            f"Servo input = {current_input:.2f}  |  "
            f"Flap angle = {res['flap_angle_deg'][current_idx]:.2f}°  |  "
            f"Rod length = {rod_length:.3f}"
        )
    else:
        angle_text = f"Servo input = {current_input:.2f}  |  No valid geometry  |  Rod length = {rod_length:.3f}"

    return physical, curve, angle_text


def _build_physical_figure(valid, theta, attach_x, attach_y,
                           servo_x, servo_y, servo_travel, current_input,
                           flap_x, flap_y):
    fig = go.Figure()

    flap_len = max(np.hypot(flap_x, flap_y) * 1.5, 2.0)
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
        xaxis={"title": "x"},
        yaxis={"title": "y", "scaleanchor": "x", "scaleratio": 1},
        showlegend=True,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def _build_curve_figure(res, current_idx, servo_travel):
    x = res["servo_input"]
    y = res["flap_angle_deg"]
    valid = res["valid"]
    ratio = torque_force_ratio(res, servo_travel)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": "blue", "width": 2},
            name="Flap angle",
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ratio,
            mode="lines",
            line={"color": "orange", "width": 2},
            name="Torque / servo force",
            yaxis="y2",
            connectgaps=False,
        )
    )
    if valid[current_idx]:
        fig.add_trace(
            go.Scatter(
                x=[x[current_idx]],
                y=[y[current_idx]],
                mode="markers",
                marker={"color": "red", "size": 12},
                name="Current input",
            )
        )
        if np.isfinite(ratio[current_idx]):
            fig.add_trace(
                go.Scatter(
                    x=[x[current_idx]],
                    y=[ratio[current_idx]],
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
               "tickfont": {"color": "blue"}},
        yaxis2={
            "title": {"text": "Torque / servo force (length)", "font": {"color": "orange"}},
            "tickfont": {"color": "orange"},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 40, "r": 60, "t": 60, "b": 40},
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
