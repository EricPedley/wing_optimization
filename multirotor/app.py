"""Interactive Dash app for the quad propulsion design model.

Same shape as linkage/app.py: sliders are the design's *current* geometry
and, for the design variables, the anchor that a "<=" / ">=" constraint mode
is measured against; the optimizer (multirotor.fastopt) runs automatically
whenever an input settles and its result is held separately from the
sliders, applied to them only on request.

What is specific to this model, versus the linkage app: the five design
variables (kV, stator volume, prop diameter, blade count, pitch) don't by
themselves have a "good" direction the way linkage geometry does, so instead
of a single fixed objective this lets the user choose which one of four
output quantities (TWR, current draw, spin-up time, tip Mach) to optimize,
and turn the other three on or off as constraints with adjustable
thresholds -- this is the "tweak constraints and objective" part. The
"tweak assumptions" part is the four physical-assumption sliders (battery
voltage, other-component mass, and the two aerodynamic constants this
session's bench-data calibration left with the widest uncertainty bands).
"""

import threading

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update
from plotly.subplots import make_subplots

import multirotor.fastopt as fo
import multirotor.quad_model as qm

DESIGN_PARAMS = [
    {"id": "kv", "name": "Motor kV (rpm/V)", "min": 3000, "max": 30000, "step": 50, "value": 18000},
    {"id": "stator-volume-mm3", "name": "Stator volume (mm³)", "min": 150, "max": 800, "step": 5, "value": 300},
    {"id": "prop-diameter-m", "name": "Prop diameter (m)", "min": 0.03, "max": 0.09, "step": 0.001, "value": 0.05},
    {"id": "blade-count", "name": "Blade count", "min": 2, "max": 5, "step": 0.1, "value": 3},
    {"id": "pitch-m", "name": "Pitch (m)", "min": 0.01, "max": 0.08, "step": 0.001, "value": 0.03},
]

ASSUMPTION_PARAMS = [
    {"id": "vbat", "name": "Battery voltage (V)", "min": 1.0, "max": 16.8, "step": 0.1, "value": qm.VBAT},
    {"id": "other-mass", "name": "Everything-else mass (g)", "min": 10, "max": 150, "step": 1,
     "value": qm.OTHER_MASS_KG * 1e3},
    {"id": "cl-alpha", "name": "Prop lift slope, CL_ALPHA (1/rad)", "min": 1.0, "max": 8.0, "step": 0.05,
     "value": fo.ASSUMPTION_DEFAULTS["cl_alpha"]},
    {"id": "induced-power-factor", "name": "Induced power factor κ", "min": 1.0, "max": 3.0, "step": 0.05,
     "value": fo.ASSUMPTION_DEFAULTS["induced_power_factor"]},
]

CONSTRAINT_OPTIONS = [
    {"label": "= (locked)", "value": "fixed"},
    {"label": "≤ slider", "value": "le"},
    {"label": "≥ slider", "value": "ge"},
    {"label": "free", "value": "free"},
]

OBJECTIVE_OPTIONS = [
    {"label": "No optimization", "value": "none"},
    {"label": "Maximize thrust-to-weight ratio", "value": "twr"},
    {"label": "Minimize current at full throttle", "value": "current_a"},
    {"label": "Minimize spin-up time (10%→90% throttle)", "value": "spin_up_s"},
    {"label": "Minimize tip Mach at full throttle", "value": "tip_mach"},
]

QUANTITIES = fo.QUANTITIES
N_SWEEP = 41


def _var_name(param_id):
    return param_id.replace("-", "_")


def _design_slider(p):
    row = dcc.Slider(
        id=p["id"], min=p["min"], max=p["max"], step=p["step"], value=p["value"],
        tooltip={"placement": "bottom", "always_visible": False},
    )
    return html.Div(
        [
            html.Label(p["name"], style={"fontWeight": "bold"}),
            html.Div(
                [
                    html.Div(row, style={"flex": "1 1 auto", "minWidth": "0"}),
                    dcc.Dropdown(
                        id=f"{p['id']}-constraint", options=CONSTRAINT_OPTIONS,
                        value="free", clearable=False, style={"width": "130px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
            ),
        ],
        style={"padding": "10px"},
    )


def _assumption_slider(p):
    return html.Div(
        [
            html.Label(p["name"], style={"fontWeight": "bold"}),
            dcc.Slider(
                id=p["id"], min=p["min"], max=p["max"], step=p["step"], value=p["value"],
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        style={"padding": "10px"},
    )


def _quantity_row(q):
    label = fo.QUANTITY_LABELS[q]
    default_threshold = fo.QUANTITY_DEFAULT_THRESHOLD[q]
    sense = fo.QUANTITY_SENSE[q]
    slider_max = default_threshold * 3 if default_threshold > 0.5 else 1.0
    return html.Div(
        [
            dcc.Checklist(
                id=f"{q}-enable",
                options=[{"label": f" Constrain: {label} {'≥' if sense == 'max' else '≤'}",
                          "value": "on"}],
                value=["on"],
            ),
            dcc.Slider(
                id=f"{q}-threshold", min=0.0, max=slider_max,
                step=slider_max / 200, value=default_threshold,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id=f"{q}-current-value", style={"fontSize": "0.85em", "color": "#666"}),
        ],
        id=f"{q}-row",
        style={"padding": "8px 0", "borderBottom": "1px solid #eee"},
    )


app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Quad Propulsion Design"),
        html.P(
            "Free variables: motor kV, stator volume, propeller diameter, blade count, "
            "and pitch. Pick which output to optimize, which to hold as a constraint, "
            "and tweak the assumptions below to see how the design point moves.",
            style={"color": "#555"},
        ),
        html.Div(
            [
                html.Div(
                    [html.H3("Design variables"),
                     html.Div([_design_slider(p) for p in DESIGN_PARAMS],
                               style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                      "gap": "6px"})],
                ),
                html.Div(
                    [html.H3("Physical assumptions"),
                     html.Div([_assumption_slider(p) for p in ASSUMPTION_PARAMS],
                               style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                      "gap": "6px"})],
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "3fr 2fr", "gap": "20px",
                   "padding": "10px", "border": "1px solid #ddd", "borderRadius": "6px",
                   "margin": "10px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Objective", style={"fontWeight": "bold"}),
                        dcc.Dropdown(id="objective", options=OBJECTIVE_OPTIONS,
                                     value="twr", clearable=False),
                        html.Button(
                            "Apply result to sliders", id="apply-result", n_clicks=0,
                            style={"marginTop": "10px", "padding": "8px 14px",
                                   "fontSize": "0.95em", "width": "100%"},
                        ),
                    ],
                    style={"flex": "1 1 260px"},
                ),
                html.Div(
                    [html.Label("Constraints", style={"fontWeight": "bold"})]
                    + [_quantity_row(q) for q in QUANTITIES],
                    style={"flex": "2 1 480px"},
                ),
                dcc.Loading(html.Div(id="optimize-status"), type="dot", delay_show=150),
            ],
            style={"display": "flex", "gap": "24px", "padding": "10px",
                   "border": "1px solid #ddd", "borderRadius": "6px", "margin": "10px",
                   "flexWrap": "wrap"},
        ),
        dcc.Graph(id="sweep-graph", style={"height": "650px"}),
        dcc.Store(id="design"),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto"},
)


# --- Objective/constraint row interlock --------------------------------------
#
# A quantity being optimized shouldn't also be shown as an editable
# constraint on itself -- disable (not hide, so the layout doesn't jump)
# whichever row matches the current objective.


@callback(
    [Output(f"{q}-enable", "value") for q in QUANTITIES]
    + [Output(f"{q}-threshold", "disabled") for q in QUANTITIES]
    + [Output(f"{q}-row", "style") for q in QUANTITIES],
    Input("objective", "value"),
    [State(f"{q}-enable", "value") for q in QUANTITIES],
)
def sync_objective_rows(objective, *enable_states):
    enable_out, disabled_out, style_out = [], [], []
    base_style = {"padding": "8px 0", "borderBottom": "1px solid #eee"}
    for q, enabled in zip(QUANTITIES, enable_states):
        is_objective = (q == objective)
        enable_out.append([] if is_objective else (enabled or []))
        disabled_out.append(is_objective)
        style_out.append({**base_style, "opacity": "0.4" if is_objective else "1.0"})
    return enable_out + disabled_out + style_out


# --- Run the optimizer whenever an input settles ------------------------------


@callback(
    Output("design", "data"),
    Output("optimize-status", "children"),
    [Input(p["id"], "value") for p in DESIGN_PARAMS],
    [Input(f"{p['id']}-constraint", "value") for p in DESIGN_PARAMS],
    [Input(p["id"], "value") for p in ASSUMPTION_PARAMS],
    Input("objective", "value"),
    [Input(f"{q}-enable", "value") for q in QUANTITIES],
    [Input(f"{q}-threshold", "value") for q in QUANTITIES],
)
def run_optimization(*state):
    n = len(DESIGN_PARAMS)
    slider_values = state[:n]
    constraint_modes = state[n:2 * n]
    a = 2 * n
    assumption_values = state[a:a + len(ASSUMPTION_PARAMS)]
    b = a + len(ASSUMPTION_PARAMS)
    objective = state[b]
    nq = len(QUANTITIES)
    enable_values = state[b + 1:b + 1 + nq]
    threshold_values = state[b + 1 + nq:b + 1 + 2 * nq]

    values = {_var_name(p["id"]): float(v) for p, v in zip(DESIGN_PARAMS, slider_values)}
    modes = {_var_name(p["id"]): m for p, m in zip(DESIGN_PARAMS, constraint_modes)}
    ranges = {_var_name(p["id"]): (float(p["min"]), float(p["max"])) for p in DESIGN_PARAMS}

    assumptions = {
        "vbat": float(assumption_values[0]),
        "other_mass_kg": float(assumption_values[1]) / 1e3,
        "cl_alpha": float(assumption_values[2]),
        "induced_power_factor": float(assumption_values[3]),
    }
    enabled = {q: bool(e) for q, e in zip(QUANTITIES, enable_values)}
    thresholds = {q: float(t) for q, t in zip(QUANTITIES, threshold_values)}

    if objective == "none":
        design = {"values": values, "assumptions": assumptions, "optimized": False}
        return design, html.Div("Optimization off — showing the slider design.",
                                 style={"color": "#666"})

    result = fo.optimize(values, modes, ranges, objective, enabled, thresholds, assumptions)
    design = {"values": result["values"], "assumptions": assumptions, "optimized": True}
    return design, _status(result, objective)


def _status(result, objective):
    start, best = result["start_metrics"], result["best_metrics"]
    lines = [
        html.Div(f"{result['message']}  ({result['elapsed'] * 1000:.0f} ms, "
                 f"{result['n_free']} free variable(s))", style={"fontWeight": "bold"}),
    ]
    if objective in fo.QUANTITY_LABELS:
        label = fo.QUANTITY_LABELS[objective]
        lines.append(html.Div(f"{label}: {start[objective]:.4g} → {best[objective]:.4g}"))
    for q in QUANTITIES:
        if q == objective:
            continue
        lines.append(html.Div(f"{fo.QUANTITY_LABELS[q]}: {best[q]:.4g}",
                               style={"color": "#666", "fontSize": "0.9em"}))
    lines.append(html.Div(
        "  ".join(f"{p['name'].split(' (')[0]}={best_v:.4g}"
                  for p, best_v in zip(DESIGN_PARAMS,
                                       (result["values"][_var_name(p["id"])]
                                        for p in DESIGN_PARAMS))),
        style={"paddingTop": "6px", "fontFamily": "monospace", "fontSize": "0.85em"},
    ))
    return lines


@callback(
    [Output(p["id"], "value") for p in DESIGN_PARAMS],
    Input("apply-result", "n_clicks"),
    State("design", "data"),
    prevent_initial_call=True,
)
def apply_result(_n_clicks, design):
    if not design or not design.get("optimized"):
        return [no_update] * len(DESIGN_PARAMS)
    values = design["values"]
    return [round(values[_var_name(p["id"])], 6) for p in DESIGN_PARAMS]


# --- Live current-value readouts on each constraint row -----------------------


@callback(
    [Output(f"{q}-current-value", "children") for q in QUANTITIES],
    Input("design", "data"),
)
def update_quantity_readouts(design):
    if not design:
        return [no_update] * len(QUANTITIES)
    x = np.array([design["values"][k] for k in fo.DESIGN_VARS])
    r = fo.evaluate_configurable(x, design["assumptions"]["vbat"],
                                  design["assumptions"]["other_mass_kg"],
                                  design["assumptions"]["cl_alpha"],
                                  design["assumptions"]["induced_power_factor"])
    return [f"current: {float(r[q]):.4g}" for q in QUANTITIES]


# --- Throttle-sweep plot -------------------------------------------------------


@callback(
    Output("sweep-graph", "figure"),
    Input("design", "data"),
)
def update_sweep(design):
    if not design:
        return no_update
    v = design["values"]
    a = design["assumptions"]
    x = np.array([v[k] for k in fo.DESIGN_VARS])

    throttles = np.linspace(0.05, 1.0, N_SWEEP)
    s = qm.throttle_sweep(x, throttles, vbat=a["vbat"], other_mass_kg=a["other_mass_kg"],
                           cl_alpha=a["cl_alpha"], induced_power_factor=a["induced_power_factor"])

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Current per motor (A)", "Thrust, all 4 motors (g)",
        "RPM", "Tip Mach"))

    volts = throttles * a["vbat"]
    thrust_total_g = np.asarray(s["thrust_n"]) * 4.0 / 9.81 * 1e3

    fig.add_trace(go.Scatter(x=volts, y=np.asarray(s["current_a"]), mode="lines",
                              line={"color": "crimson"}, name="Current"), row=1, col=1)
    fig.add_trace(go.Scatter(x=volts, y=thrust_total_g, mode="lines",
                              line={"color": "green"}, name="Thrust"), row=1, col=2)
    fig.add_trace(go.Scatter(x=volts, y=np.asarray(s["rpm"]), mode="lines",
                              line={"color": "blue"}, name="RPM"), row=2, col=1)
    fig.add_trace(go.Scatter(x=volts, y=np.asarray(s["tip_mach"]), mode="lines",
                              line={"color": "darkorange"}, name="Tip Mach"), row=2, col=2)

    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(title_text="Volts applied", row=row, col=col)

    fig.update_layout(
        title=f"Throttle sweep — {'optimized' if design.get('optimized') else 'slider'} design "
              f"(battery {a['vbat']:.1f}V)",
        showlegend=False, margin={"l": 50, "r": 30, "t": 80, "b": 40},
    )
    return fig


threading.Thread(target=fo.warmup, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True, port=8051)
