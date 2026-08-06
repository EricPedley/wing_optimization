"""Simple geometry plots for one design point.

Four views, each answering one question that is hard to see from numbers:

    planform  -- what the wing looks like from above, where the elevons and
                 motors sit, and how much of the elevon the props actually blow
    section   -- the root section with the battery and servo drawn in, which is
                 the packaging constraint that has been driving the design
    fit       -- how much section depth is available at each chord station
                 versus what the servo needs, so the feasible band is visible
    authority -- control moment in each flight regime, since the regimes differ
                 by an order of magnitude and the sizing case is not cruise

Deliberately schematic.  The section is a crude thickness profile, not a real
airfoil, because the airfoil shape has not been chosen yet -- what is being
checked here is whether the things inside the wing fit.

Every figure takes a design vector, defaulting to the optimizer's result when
one has been cached and to the hand-picked baseline otherwise.

    uv run python -m airfoil.plots              # optimum if available
    uv run python -m airfoil.plots --baseline   # the hand-picked point
    uv run python -m airfoil.plots --optimize   # re-run the optimizer first
"""

import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import airfoil.airfoil_model as am
import airfoil.optimize as opt


def default_design():
    """The optimizer's result if it has been run, else the baseline.

    Falling back rather than failing keeps the plots usable before the
    optimizer has ever been run, which is also the case where someone is most
    likely to be poking at the model by hand.
    """
    x = opt.load_optimum()
    return am.BASELINE if x is None else x


def _section_outline(chord, max_thickness, n=80):
    """Upper and lower surface of the schematic section, in metres.

    Symmetric about the chord line: camber is not chosen yet, and the packaging
    question this drawing answers does not depend on it.
    """
    x = np.linspace(0.0, 1.0, n)
    half = 0.5 * np.asarray(am.thickness_at(x, max_thickness))
    return x * chord, half


def planform_figure(p=None):
    """Top view: chords, elevon, motors, and slipstream footprint."""
    p = default_design() if p is None else p
    g = am.unpack(p)
    r = am.evaluate(p)

    root_c = float(g["root_chord"])
    tip_c = float(g["tip_chord"])
    semi = 0.5 * am.SPAN
    x_hinge = float(g["x_hinge"])
    ein = float(g["elevon_inboard_y"])
    eout = float(g["elevon_outboard_y"])
    motor_y = float(g["motor_y"])

    # x runs aft from the leading edge, y outboard from the centreline.  Leading
    # edge is straight, so all the taper shows up at the trailing edge.
    def chord_at(y):
        return root_c + (tip_c - root_c) * (abs(y) / semi)

    fig = go.Figure()

    ys = np.linspace(-semi, semi, 200)
    cs = np.array([chord_at(y) for y in ys])

    fig.add_trace(go.Scatter(
        x=np.concatenate([ys, ys[::-1]]) * 1e3,
        y=np.concatenate([np.zeros_like(ys), cs[::-1]]) * 1e3,
        fill="toself", fillcolor="rgba(200,200,200,0.35)",
        line={"color": "black", "width": 2},
        name="Wing", hoverinfo="skip",
    ))

    # Hinge line, drawn only across the elevon span where it exists.
    for sign in (1, -1):
        ye = np.linspace(sign * ein, sign * eout, 40)
        fig.add_trace(go.Scatter(
            x=ye * 1e3,
            y=np.array([chord_at(y) * x_hinge for y in ye]) * 1e3,
            mode="lines", line={"color": "red", "width": 2, "dash": "dash"},
            name="Hinge line" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    # Elevon area, shaded so its span and chordwise extent are both obvious.
    for sign in (1, -1):
        ye = np.linspace(sign * ein, sign * eout, 40)
        hinge = np.array([chord_at(y) * x_hinge for y in ye])
        trail = np.array([chord_at(y) for y in ye])
        fig.add_trace(go.Scatter(
            x=np.concatenate([ye, ye[::-1]]) * 1e3,
            y=np.concatenate([hinge, trail[::-1]]) * 1e3,
            fill="toself", fillcolor="rgba(220,80,80,0.35)",
            line={"color": "rgba(0,0,0,0)"},
            name="Elevon" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    # Slipstream footprint: the strip of span the prop actually blows.  Its
    # overlap with the elevon is what sets control authority in hover.
    half_w = 0.5 * am.SLIPSTREAM_WIDTH_FACTOR * am.PROP_DIAMETER
    for sign in (1, -1):
        lo, hi = sign * motor_y - half_w, sign * motor_y + half_w
        fig.add_trace(go.Scatter(
            x=np.array([lo, hi, hi, lo, lo]) * 1e3,
            y=np.array([0, 0, root_c, root_c, 0]) * 1e3,
            mode="lines", line={"color": "blue", "width": 1, "dash": "dot"},
            fill="toself", fillcolor="rgba(80,120,220,0.15)",
            name="Slipstream" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    # Battery on the centreline and servos outboard, which is the layout that
    # lets both fit: they occupy different spanwise stations rather than
    # competing for the same chord.
    fig.add_trace(go.Scatter(
        x=np.array([-0.5 * am.SERVO_WIDTH, 0.5 * am.SERVO_WIDTH,
                    0.5 * am.SERVO_WIDTH, -0.5 * am.SERVO_WIDTH,
                    -0.5 * am.SERVO_WIDTH]) * 1e3,
        y=np.array([am.BATTERY_STATION * root_c,
                    am.BATTERY_STATION * root_c,
                    am.BATTERY_STATION * root_c + am.BATTERY_LENGTH,
                    am.BATTERY_STATION * root_c + am.BATTERY_LENGTH,
                    am.BATTERY_STATION * root_c]) * 1e3,
        mode="lines", line={"color": "green", "width": 2},
        fill="toself", fillcolor="rgba(80,180,80,0.35)",
        name="Battery", hoverinfo="skip",
    ))

    servo_y = float(g["servo_y"])
    servo_c = float(g["servo_chord"])
    station = float(g["servo_station"])
    for sign in (1, -1):
        y0 = sign * servo_y - 0.5 * am.SERVO_WIDTH
        y1 = sign * servo_y + 0.5 * am.SERVO_WIDTH
        x0 = station * servo_c - 0.5 * am.SERVO_LENGTH
        x1 = station * servo_c + 0.5 * am.SERVO_LENGTH
        fig.add_trace(go.Scatter(
            x=np.array([y0, y1, y1, y0, y0]) * 1e3,
            y=np.array([x0, x0, x1, x1, x0]) * 1e3,
            mode="lines", line={"color": "purple", "width": 2},
            fill="toself", fillcolor="rgba(160,80,200,0.4)",
            name="Servo" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    # Props, drawn at the leading edge where they actually mount.
    theta = np.linspace(0, 2 * np.pi, 60)
    for sign in (1, -1):
        fig.add_trace(go.Scatter(
            x=(sign * motor_y + 0.5 * am.PROP_DIAMETER * np.cos(theta)) * 1e3,
            y=(0.5 * am.PROP_DIAMETER * np.sin(theta)) * 1e3,
            mode="lines", line={"color": "blue", "width": 2},
            name="Prop disc" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    fig.update_layout(
        title=(f"Planform  --  {float(r['area']) * 1e4:.0f} cm^2,"
               f" AR {float(r['aspect_ratio']):.2f},"
               f" {float(r['wash_fraction']) * 100:.0f}% of elevon blown"),
        xaxis={"title": "span, mm"},
        # Aft is down: the wing is drawn as seen from above with the leading
        # edge at the top, which is how the planform is normally read.
        yaxis={"title": "chord, mm", "scaleanchor": "x", "scaleratio": 1,
               "autorange": "reversed"},
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    return fig


def section_figure(p=None):
    """Root section with the battery and servo drawn to scale inside it."""
    p = default_design() if p is None else p
    g = am.unpack(p)
    r = am.evaluate(p)

    chord = float(g["root_chord"])
    thick = float(g["root_thickness"])
    station = float(g["servo_station"])
    x_hinge = float(g["x_hinge"])

    xs, half = _section_outline(chord, thick)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]) * 1e3,
        y=np.concatenate([half, -half[::-1]]) * 1e3,
        fill="toself", fillcolor="rgba(200,200,200,0.35)",
        line={"color": "black", "width": 2},
        name="Section", hoverinfo="skip",
    ))

    # Battery: sits forward, where the section is deepest, which is also where
    # it needs to be for the centre of gravity.
    batt_x0 = am.BATTERY_STATION * chord
    batt_x1 = batt_x0 + am.BATTERY_LENGTH
    batt_h = 0.5 * am.MIN_ROOT_THICKNESS
    fig.add_trace(go.Scatter(
        x=np.array([batt_x0, batt_x1, batt_x1, batt_x0, batt_x0]) * 1e3,
        y=np.array([-batt_h, -batt_h, batt_h, batt_h, -batt_h]) * 1e3,
        mode="lines", line={"color": "green", "width": 2},
        fill="toself", fillcolor="rgba(80,180,80,0.3)",
        name="Battery", hoverinfo="skip",
    ))

    # Servo, drawn dashed because it does not live in this section: it sits
    # outboard near its elevon, where the battery is not in the way.  Shown here
    # at its own local chord so the two can be compared, which is the whole
    # reason it has to go outboard.
    servo_chord = float(g["servo_chord"])
    sv_x0 = station * servo_chord - 0.5 * am.SERVO_LENGTH
    sv_x1 = station * servo_chord + 0.5 * am.SERVO_LENGTH
    sv_h = 0.5 * am.SERVO_DEPTH
    fig.add_trace(go.Scatter(
        x=np.array([sv_x0, sv_x1, sv_x1, sv_x0, sv_x0]) * 1e3,
        y=np.array([-sv_h, -sv_h, sv_h, sv_h, -sv_h]) * 1e3,
        mode="lines", line={"color": "purple", "width": 2, "dash": "dash"},
        fill="toself", fillcolor="rgba(160,80,200,0.25)",
        name=f"Servo (at {float(g['servo_span_frac']) * 100:.0f}% semi-span)",
        hoverinfo="skip",
    ))

    # Pushrod from the servo output to the hinge, straight-line schematic.
    fig.add_trace(go.Scatter(
        x=np.array([sv_x1, x_hinge * servo_chord]) * 1e3,
        y=np.array([0.0, 0.0]) * 1e3,
        mode="lines", line={"color": "purple", "width": 1, "dash": "dot"},
        name="Pushrod", hoverinfo="skip",
    ))

    # Hinge line through the full local thickness.
    h_half = float(am.thickness_at(x_hinge, thick)) * 0.5
    fig.add_trace(go.Scatter(
        x=np.array([x_hinge * chord, x_hinge * chord]) * 1e3,
        y=np.array([-h_half, h_half]) * 1e3,
        mode="lines+markers", line={"color": "red", "width": 2, "dash": "dash"},
        marker={"color": "red", "size": 6}, name="Hinge",
    ))

    slack = float(r["volume_slack"]) * 1e3
    fig.update_layout(
        title=(f"Root section  --  {chord * 1e3:.0f} mm chord,"
               f" {float(r['root_tc']) * 100:.1f}% thick,"
               f" slack {slack:+.1f} mm"),
        xaxis={"title": "chord, mm"},
        yaxis={"title": "thickness, mm", "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    return fig


def fit_figure(p=None):
    """Depth available along the chord versus what the servo needs.

    The band where the curve clears the servo line is the set of chord stations
    the servo can actually sit at, which is the constraint that has shaped the
    whole centre section.
    """
    p = default_design() if p is None else p
    g = am.unpack(p)
    # The servo's own section, not the root: it sits outboard, where the wing is
    # both shorter in chord and thinner, so the root would flatter it.
    thick = float(g["servo_thickness"])
    station = float(g["servo_station"])

    x = np.linspace(0.02, 0.95, 200)
    depth = np.asarray(am.thickness_at(x, thick))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x * 100, y=depth * 1e3, mode="lines",
        line={"color": "black", "width": 2}, name="Depth available",
    ))
    fig.add_trace(go.Scatter(
        x=x * 100, y=np.full_like(x, am.SERVO_DEPTH * 1e3), mode="lines",
        line={"color": "purple", "width": 2, "dash": "dash"},
        name="Servo needs",
    ))

    feasible = depth >= am.SERVO_DEPTH
    if feasible.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([x[feasible], x[feasible][::-1]]) * 100,
            y=np.concatenate([depth[feasible],
                              np.full(feasible.sum(), am.SERVO_DEPTH)[::-1]]) * 1e3,
            fill="toself", fillcolor="rgba(80,180,80,0.25)",
            line={"color": "rgba(0,0,0,0)"}, name="Servo fits",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=[station * 100], y=[float(am.thickness_at(station, thick)) * 1e3],
        mode="markers", marker={"color": "purple", "size": 12},
        name="Chosen station",
    ))
    fig.add_trace(go.Scatter(
        x=[float(g["x_hinge"]) * 100, float(g["x_hinge"]) * 100],
        y=[0.0, thick * 1e3], mode="lines",
        line={"color": "red", "width": 2, "dash": "dash"}, name="Hinge",
    ))

    lo = x[feasible].min() * 100 if feasible.any() else float("nan")
    hi = x[feasible].max() * 100 if feasible.any() else float("nan")
    fig.update_layout(
        title=f"Servo fit  --  feasible from {lo:.0f}% to {hi:.0f}% chord",
        xaxis={"title": "chord station, %"},
        yaxis={"title": "section depth, mm", "rangemode": "tozero"},
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    return fig


def authority_figure(p=None, deflection_deg=10.0):
    """Control moment by flight regime, on a log axis.

    Log scale because hover and cruise differ by roughly an order of magnitude,
    and the point of the plot is that the sizing case is whichever regime is
    lowest, not the one that is easiest to think about.
    """
    p = default_design() if p is None else p
    r = am.evaluate(p, deflection_deg=deflection_deg)
    a = r["authority"]
    names = list(a.keys())

    fig = go.Figure()
    for key, colour in (("pitch", "rgba(220,80,80,0.85)"),
                        ("roll", "rgba(80,120,220,0.85)")):
        fig.add_trace(go.Bar(
            x=names,
            y=[abs(float(a[n][key])) * 1e3 for n in names],
            name=key.capitalize(), marker_color=colour,
        ))

    fig.update_layout(
        title=f"Control authority at {deflection_deg:.0f} deg deflection",
        xaxis={"title": "flight regime"},
        yaxis={"title": "moment, mN.m", "type": "log"},
        barmode="group",
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    return fig


def dashboard(p=None, label=None):
    """All four views in one figure."""
    p = default_design() if p is None else p
    figs = [planform_figure(p), section_figure(p),
            fit_figure(p), authority_figure(p)]

    combined = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f.layout.title.text for f in figs],
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )
    for i, f in enumerate(figs):
        row, col = i // 2 + 1, i % 2 + 1
        for trace in f.data:
            combined.add_trace(trace, row=row, col=col)
        combined.update_xaxes(title_text=f.layout.xaxis.title.text,
                              row=row, col=col)
        combined.update_yaxes(title_text=f.layout.yaxis.title.text,
                              row=row, col=col)

    # Carry over the axis settings that make each view readable: true aspect
    # for the two geometry plots, log scale for the authority bars.
    combined.update_yaxes(scaleanchor="x", scaleratio=1,
                          autorange="reversed", row=1, col=1)
    combined.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    combined.update_yaxes(type="log", row=2, col=2)
    r = am.evaluate(p)
    title = "Tailsitter geometry"
    if label:
        title = f"{title}  --  {label}"
    title += (f"  --  stall {float(r['v_stall']):.2f} m/s,"
              f" {float(r['mass']) * 1e3:.1f} g, TWR {float(r['twr']):.2f}")
    combined.update_layout(
        height=1000, showlegend=True, barmode="group",
        title_text=title,
        margin={"l": 60, "r": 40, "t": 80, "b": 40},
    )
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true",
                        help="plot the hand-picked baseline instead of the optimum")
    parser.add_argument("--optimize", action="store_true",
                        help="re-run the optimizer before plotting")
    parser.add_argument("-o", "--out", default="airfoil/plots.html")
    args = parser.parse_args()

    if args.optimize:
        print("Running optimizer...")
        result = opt.optimize()
        print(f"  converged in {result['elapsed']:.1f} s"
              f"  (cost {result['cost']:.4f})")

    if args.baseline:
        p, label = am.BASELINE, "hand-picked baseline"
    else:
        x = opt.load_optimum()
        if x is None:
            p, label = am.BASELINE, "hand-picked baseline (no optimum cached)"
            print("No cached optimum; run with --optimize to generate one.")
        else:
            p, label = x, "optimized"

    dashboard(p, label=label).write_html(args.out, include_plotlyjs="cdn")
    print(f"wrote {args.out}  ({label})")


if __name__ == "__main__":
    main()
