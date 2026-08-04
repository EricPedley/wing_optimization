"""Interactive Plotly Dash app for the flap-servo simulation."""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

from core import simulate_flap

PARAMS = [
    {"id": "servo-x", "name": "Servo start x", "min": 0, "max": 20, "step": 0.1, "value": 5.0},
    {"id": "servo-y", "name": "Servo y", "min": 0, "max": 20, "step": 0.1, "value": 8.0},
    {"id": "servo-travel", "name": "Servo travel x", "min": 5, "max": 15, "step": 0.1, "value": 5.0},
    {"id": "flap-x", "name": "Flap attach x", "min": 0, "max": 40, "step": 0.1, "value": 5.0},
    {"id": "flap-y", "name": "Flap attach y", "min": 5, "max": 30, "step": 0.1, "value": 5.0},
    {"id": "rod-length", "name": "Rod length", "min": 10, "max": 40, "step": 0.05, "value": 10.44},
    {"id": "current-input", "name": "Servo input", "min": 0, "max": 1, "step": 0.01, "value": 0.0},
]


def _slider(p):
    return html.Div(
        [
            html.Label(p["name"], style={"fontWeight": "bold"}),
            dcc.Slider(
                id=p["id"],
                min=p["min"],
                max=p["max"],
                step=p["step"],
                value=p["value"],
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        style={"padding": "10px"},
    )


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
    Output("physical-graph", "figure"),
    Output("curve-graph", "figure"),
    Output("angle-display", "children"),
    [Input(p["id"], "value") for p in PARAMS],
)
def update(servo_x, servo_y, servo_travel, flap_x, flap_y, rod_length, current_input):
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
    curve = _build_curve_figure(res, current_idx)

    if valid:
        angle_text = f"Servo input = {current_input:.2f}  |  Flap angle = {res['flap_angle_deg'][current_idx]:.2f}°"
    else:
        angle_text = f"Servo input = {current_input:.2f}  |  No valid geometry for these parameters"

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


def _build_curve_figure(res, current_idx):
    x = res["servo_input"]
    y = res["flap_angle_deg"]
    valid = res["valid"]

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

    fig.update_layout(
        title="Servo input vs. flap angle",
        xaxis={"title": "Servo input (0 → 1)", "range": [0, 1]},
        yaxis={"title": "Flap angle (degrees)"},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
