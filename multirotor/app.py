"""Interactive Dash app for the quad propulsion design model.

Same shape as linkage/app.py: sliders are the design's *current* geometry
and, for the design variables, the anchor that a "<=" / ">=" constraint mode
is measured against; the optimizer (multirotor.fastopt) runs automatically
whenever an input settles and its result is held separately from the
sliders, applied to them only on request.

What is specific to this model, versus the linkage app:

- The four continuous design variables (kV, stator volume, prop diameter,
  pitch) don't by themselves have a "good" direction the way linkage
  geometry does, so instead of a single fixed objective this lets the user
  choose which one of four output quantities (TWR, current draw, spin-up
  time, tip Mach) to optimize, and turn the other three on or off as
  constraints with adjustable thresholds.
- Blade count is physically discrete (2 or 3 blades, not 2.4), so it is not
  a continuous slider at all: every optimizer run is two full rollouts, one
  with blade count fixed at 2 and one at 3 (fastopt.optimize_over_blade_counts),
  and the app shows whichever converged to the better cost.
- Prop diameter is shown in mm and pitch in inches (the units these are
  actually specified in), converted to/from the model's native SI units
  (metres) at the UI boundary; prop diameter is capped at 3 inches, a
  packaging limit for the frame this is being sized for.
- "Tweak assumptions" covers battery voltage, other-component mass, and the
  two aerodynamic constants this session's bench-data calibration left with
  the widest uncertainty bands (CL_ALPHA, the induced power factor).
- A stator-volume readout also shows the two closest off-the-shelf motor
  sizes (see quad_model.REALISTIC_STATOR_SIZES) and how far off their
  volumes are, since stator_volume_mm3 is continuous but motors are not.
- A hover flight-time estimate (quad_model.hover_point) for a
  user-adjustable battery capacity, since TWR and the current constraint are
  both evaluated at full throttle and say nothing about endurance.
"""

import threading

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update
from plotly.subplots import make_subplots

import multirotor.battery_model as bm
import multirotor.fastopt as fo
import multirotor.quad_model as qm

MM_PER_M = 1e3
IN_PER_M = 1.0 / 25.4e-3

# Each design param carries its own display unit via si_scale: the slider
# shows value in that unit, and value * si_scale is what the model (which
# works in kV, mm^3, and SI metres) actually receives. "var" is the
# quad_model.DESIGN_VARS key it maps to -- kept explicit rather than derived
# from the slider id, so a UI id can read naturally (e.g. "prop-diameter")
# without having to spell out the backend variable name.
DESIGN_PARAMS = [
    {"id": "kv", "var": "kv", "name": "Motor kV (rpm/V)",
     "min": 3000, "max": 30000, "step": 50, "value": 18000, "si_scale": 1.0},
    {"id": "stator-volume", "var": "stator_volume_mm3", "name": "Stator volume (mm³)",
     "min": 150, "max": 800, "step": 5, "value": 300, "si_scale": 1.0},
    {"id": "prop-diameter", "var": "prop_diameter_m", "name": "Prop diameter (mm)",
     "min": 20.0, "max": qm.MAX_PROP_DIAMETER_M * MM_PER_M, "step": 0.5, "value": 50.0,
     "si_scale": 1.0 / MM_PER_M},
    {"id": "pitch", "var": "pitch_m", "name": "Pitch (in)",
     "min": 0.4, "max": 3.2, "step": 0.05, "value": 1.2, "si_scale": 1.0 / IN_PER_M},
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

BATTERY_PARAM = {"id": "battery-name", "name": "Battery",
                  "options": [{"label": f"BetaFPV LAVA II 1S {n} "
                                        f"({bm.mass_kg(n) * 1e3:.1f}g)", "value": n}
                              for n in bm.BATTERIES],
                  "value": "680mAh"}

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
    {"label": "Minimize current at a chosen throttle (TWR as a floor)",
     "value": "current_at_throttle"},
]

OBJECTIVE_THROTTLE_PARAM = {"id": "objective-throttle", "name": "Flight condition throttle (%)",
                             "min": 5, "max": 100, "step": 1,
                             "value": fo.DEFAULT_OBJECTIVE_THROTTLE_FRAC * 100}

QUANTITIES = fo.QUANTITIES
N_SWEEP = 41
# Blade count when the user has turned optimization off -- there is no
# slider for it (it is only ever chosen by the two-rollout comparison), so
# the "just show me the sliders" path needs some default to evaluate with.
DEFAULT_BLADE_COUNT_WHEN_UNOPTIMIZED = 3.0


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
            html.Div(id=f"{p['id']}-note", style={"fontSize": "0.8em", "color": "#888"}),
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
            "Free variables: motor kV, stator volume, propeller diameter, and pitch "
            "(blade count is optimized separately over 2 and 3 blades, since it's "
            "physically discrete). Pick which output to optimize, which to hold as a "
            "constraint, and tweak the assumptions below to see how the design point moves.",
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
                        html.Div(
                            [
                                html.Label(OBJECTIVE_THROTTLE_PARAM["name"],
                                           style={"fontSize": "0.85em", "color": "#666"}),
                                dcc.Slider(
                                    id=OBJECTIVE_THROTTLE_PARAM["id"],
                                    min=OBJECTIVE_THROTTLE_PARAM["min"],
                                    max=OBJECTIVE_THROTTLE_PARAM["max"],
                                    step=OBJECTIVE_THROTTLE_PARAM["step"],
                                    value=OBJECTIVE_THROTTLE_PARAM["value"],
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                html.Div(
                                    "Only used by \"Minimize current at a chosen throttle\".",
                                    style={"fontSize": "0.75em", "color": "#999"},
                                ),
                            ],
                            style={"paddingTop": "8px"},
                        ),
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
        html.Div(
            [
                html.H3("Hover flight time"),
                html.Div(
                    dcc.RadioItems(
                        id=BATTERY_PARAM["id"], options=BATTERY_PARAM["options"],
                        value=BATTERY_PARAM["value"], inline=True,
                        labelStyle={"marginRight": "16px"},
                    ),
                    style={"maxWidth": "600px"},
                ),
                html.Div(id="hover-readout", style={"paddingTop": "8px", "fontFamily": "monospace"}),
            ],
            style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "6px",
                   "margin": "10px"},
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
    Input(OBJECTIVE_THROTTLE_PARAM["id"], "value"),
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
    objective_throttle_pct = state[b + 1]
    c = b + 2
    nq = len(QUANTITIES)
    enable_values = state[c:c + nq]
    threshold_values = state[c + nq:c + 2 * nq]

    # Convert display units (mm, inches) to the model's native SI units at
    # this boundary; everything downstream of `values` is in kV/mm^3/metres.
    values = {p["var"]: float(v) * p["si_scale"] for p, v in zip(DESIGN_PARAMS, slider_values)}
    modes = {p["var"]: m for p, m in zip(DESIGN_PARAMS, constraint_modes)}
    ranges = {p["var"]: (float(p["min"]) * p["si_scale"], float(p["max"]) * p["si_scale"])
              for p in DESIGN_PARAMS}
    # blade_count has no slider; give it a placeholder so the DESIGN_VARS
    # dict is complete. optimize_over_blade_counts overrides it per rollout;
    # the "no optimization" path below overrides it with a fixed default.
    values["blade_count"] = DEFAULT_BLADE_COUNT_WHEN_UNOPTIMIZED
    modes["blade_count"] = "free"
    ranges["blade_count"] = (2.0, 3.0)

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
        return design, html.Div("Optimization off — showing the slider design "
                                 f"({int(DEFAULT_BLADE_COUNT_WHEN_UNOPTIMIZED)} blades assumed).",
                                 style={"color": "#666"})

    best, rollouts = fo.optimize_over_blade_counts(
        values, modes, ranges, objective, enabled, thresholds, assumptions,
        objective_throttle_frac=float(objective_throttle_pct) / 100.0)
    design = {"values": best["values"], "assumptions": assumptions, "optimized": True}
    return design, _status(best, rollouts, objective, enabled, thresholds, assumptions)


def _status(best, rollouts, objective, enabled, thresholds, assumptions):
    start, bm = best["start_metrics"], best["best_metrics"]
    lines = [
        html.Div(f"{best['message']}  ({best['elapsed'] * 1000:.0f} ms, "
                 f"{best['n_free']} free variable(s), "
                 f"winning blade count: {int(best['blade_count'])})",
                 style={"fontWeight": "bold"}),
        html.Div(
            "  ".join(f"{int(r['blade_count'])} blades → "
                      f"{fo.OBJECTIVE_LABELS[objective].split(',')[0]}="
                      f"{r['best_metrics'][objective]:.4g}"
                      for r in rollouts),
            style={"color": "#666", "fontSize": "0.85em"},
        ),
    ]
    if objective in fo.OBJECTIVE_LABELS:
        label = fo.OBJECTIVE_LABELS[objective]
        lines.append(html.Div(f"{label}: {start[objective]:.4g} → {bm[objective]:.4g}"))

    # --- Constraint slacks, same idea as airfoil/optimize.py's printed
    # constraint table: what the optimizer was actually held to, and whether
    # it ended up sitting exactly on a limit, comfortably inside it, or (if
    # the penalty couldn't fully satisfy it) still over.
    lines.append(html.Div("Constraints:", style={"fontWeight": "bold", "paddingTop": "6px"}))
    for q in QUANTITIES:
        if q == objective:
            lines.append(html.Div(f"  {fo.QUANTITY_LABELS[q]}: {bm[q]:.4g}  (this run's objective)",
                                   style={"color": "#666", "fontSize": "0.9em"}))
            continue
        if not enabled.get(q):
            lines.append(html.Div(f"  {fo.QUANTITY_LABELS[q]}: {bm[q]:.4g}  (not constrained)",
                                   style={"color": "#999", "fontSize": "0.9em"}))
            continue
        threshold = thresholds[q]
        sense = fo.QUANTITY_SENSE[q]
        slack = (bm[q] - threshold) if sense == "max" else (threshold - bm[q])
        limit_desc = f"{'≥' if sense == 'max' else '≤'} {threshold:.4g}"
        if slack < -1e-6:
            marker, color = f"  VIOLATED by {-slack:.4g}", "crimson"
        elif slack < 1e-3 * max(abs(threshold), 1.0):
            marker, color = "  binding (at the limit)", "darkorange"
        else:
            marker, color = f"  slack {slack:.4g}", "#2a2"
        lines.append(html.Div(f"  {fo.QUANTITY_LABELS[q]}: {bm[q]:.4g}  ({limit_desc}){marker}",
                               style={"color": color, "fontSize": "0.9em"}))

    # --- Weight breakdown ---------------------------------------------------
    unit = bm["unit"]
    other_g = assumptions["other_mass_kg"] * 1e3
    frame_g = bm["frame_mass"] * 1e3
    motors_g = unit["motor_mass"] * 4.0 * 1e3
    props_g = unit["prop_mass"] * 4.0 * 1e3
    propulsion_g = motors_g + props_g
    total_g = bm["total_mass"] * 1e3
    lines.append(html.Div("Weight breakdown:", style={"fontWeight": "bold", "paddingTop": "6px"}))
    lines.append(html.Div(
        f"  Everything else: {other_g:.2f} g   |   Frame (est.): {frame_g:.2f} g   |   "
        f"Propulsion (4x motor+prop): {propulsion_g:.2f} g   |   Total: {total_g:.2f} g",
        style={"fontSize": "0.9em"},
    ))
    lines.append(html.Div(
        f"  Propulsion split — motors: {motors_g:.2f} g ({unit['motor_mass'] * 1e3:.3f} g each)"
        f"   props: {props_g:.2f} g ({unit['prop_mass'] * 1e3:.3f} g each)",
        style={"color": "#666", "fontSize": "0.85em"},
    ))

    nearest = qm.nearest_stator_sizes(best["values"]["stator_volume_mm3"])
    lines.append(html.Div(
        "Nearest realistic stator sizes: " + ", ".join(
            f"{s['name']} ({s['volume_mm3']:.0f}mm³, {s['delta_pct']:+.1f}%)" for s in nearest),
        style={"color": "#666", "fontSize": "0.85em"},
    ))

    v = best["values"]
    lines.append(html.Div(
        f"kV={v['kv']:.4g}  Stator volume={v['stator_volume_mm3']:.4g}mm³  "
        f"Diameter={v['prop_diameter_m'] * MM_PER_M:.4g}mm  "
        f"Pitch={v['pitch_m'] * IN_PER_M:.4g}in  Blades={int(v['blade_count'])}",
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
    return [round(values[p["var"]] / p["si_scale"], 6) for p in DESIGN_PARAMS]


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


# --- Hover flight time ---------------------------------------------------------


@callback(
    Output("hover-readout", "children"),
    Input("design", "data"),
    Input(BATTERY_PARAM["id"], "value"),
)
def update_hover_readout(design, battery_name):
    if not design:
        return no_update
    x = np.array([design["values"][k] for k in fo.DESIGN_VARS])
    a = design["assumptions"]
    r = qm.hover_point(x, battery_name=battery_name, other_mass_kg=a["other_mass_kg"],
                        cl_alpha=a["cl_alpha"], induced_power_factor=a["induced_power_factor"])
    if not r["feasible"]:
        return html.Div("Cannot hover — max thrust at full throttle is below the "
                         "vehicle's weight.", style={"color": "crimson"})
    return html.Div([
        html.Div(f"Hover throttle: {r['hover_throttle_frac'] * 100:.1f}%"),
        html.Div(f"Hover current (all 4 motors): {r['hover_current_a_total']:.2f} A"),
        html.Div(f"Estimated hover flight time on {int(r['battery_mah'])} mAh "
                 f"({r['battery_name']}): {r['flight_time_min']:.1f} min",
                 style={"fontWeight": "bold"}),
        html.Div("No reserve margin included -- this is time to fully discharge (or hit "
                  "a 3.0V voltage floor) at a constant hover load as the pack sags under "
                  "load, not a safe usable flight time.",
                  style={"fontSize": "0.8em", "color": "#888"}),
    ])


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
              f"({int(v['blade_count'])} blades, battery {a['vbat']:.1f}V)",
        showlegend=False, margin={"l": 50, "r": 30, "t": 80, "b": 40},
    )
    return fig


threading.Thread(target=fo.warmup, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True, port=8051)
