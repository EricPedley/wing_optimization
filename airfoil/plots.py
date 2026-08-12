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
from pathlib import Path

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


def _section_outline(chord, max_thickness, a1=0.0, a2=0.0, n=80):
    """Upper and lower surface of the section, in metres.

    Thickness is distributed about the camber line rather than about the chord
    line, which is how a real section is built and the reason this returns two
    independent surfaces instead of one half-thickness to mirror.

    The camber line itself is returned too, because it is what the packaging
    drawing actually wants: the battery and the servo sit in the section's
    interior, so what matters is the depth available *between the surfaces* at a
    station, not the distance from either one to the chord.
    """
    x = np.linspace(0.0, 1.0, n)
    half = 0.5 * np.asarray(am.thickness_at(x, max_thickness))
    camber = np.asarray(am.camber_line(x, a1, a2)) * chord
    return x * chord, camber + half, camber - half, camber


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

    # x runs aft from the root leading edge, y outboard from the centreline.
    le_sweep = float(g["le_sweep_deg"])

    def chord_at(y):
        # Through the model's own loft, so the constant-chord centre section
        # shows up in the drawing instead of only in the numbers.
        return float(am.local_geometry(root_c, tip_c, 0.0, 0.0,
                                       abs(y) / semi)[0])

    def le_at(y):
        return float(am.leading_edge_x(abs(y) / semi, le_sweep))

    # Hinge position, computed the same way the model does it, so the drawing
    # cannot disagree with the numbers.  With a constant-chord elevon this runs
    # parallel to the trailing edge; with a constant-fraction one it converges
    # on it, which is the difference the plot is meant to make visible.
    mac = float(r["mac"])

    def hinge_at(y):
        c = chord_at(y)
        return le_at(y) + c - float(am.elevon_chord_at(c, x_hinge, mac))

    def te_at(y):
        return le_at(y) + chord_at(y)

    fig = go.Figure()

    ys = np.linspace(-semi, semi, 200)
    les = np.array([le_at(y) for y in ys])
    tes = np.array([te_at(y) for y in ys])

    fig.add_trace(go.Scatter(
        x=np.concatenate([ys, ys[::-1]]) * 1e3,
        y=np.concatenate([les, tes[::-1]]) * 1e3,
        fill="toself", fillcolor="rgba(200,200,200,0.35)",
        line={"color": "black", "width": 2},
        name="Wing", hoverinfo="skip",
    ))

    # Quarter-chord line: where section lift acts, and the sweep that actually
    # matters for a tailless aircraft.  Drawn because taper pulls it well
    # forward of the leading-edge sweep, which is easy to miss.
    fig.add_trace(go.Scatter(
        x=ys * 1e3,
        y=np.array([le_at(y) + 0.25 * chord_at(y) for y in ys]) * 1e3,
        mode="lines", line={"color": "orange", "width": 1, "dash": "dashdot"},
        name=f"c/4 ({float(r['c4_sweep_deg']):+.1f} deg)", hoverinfo="skip",
    ))

    # Centre of gravity and aerodynamic centre, and the gap between them.  This
    # is the pair that decides whether the aircraft is stable in pitch at all,
    # and it is worth drawing rather than only reporting because the reason the
    # gap is what it is -- sweep dragging the aerodynamic centre aft faster than
    # it drags the mass -- is a geometric fact that a number does not show.
    x_cg = float(r["cg_station"])
    x_ac = float(r["ac_station"])
    sm = float(r["static_margin"])
    fig.add_trace(go.Scatter(
        x=[0.0], y=[x_cg * 1e3], mode="markers",
        marker={"color": "blue", "size": 12, "symbol": "circle-cross"},
        name=f"CG ({x_cg * 1e3:.0f} mm)",
    ))
    fig.add_trace(go.Scatter(
        x=[0.0], y=[x_ac * 1e3], mode="markers",
        marker={"color": "purple", "size": 11, "symbol": "diamond"},
        name=f"AC ({x_ac * 1e3:.0f} mm),  SM {sm * 100:+.1f}%",
    ))

    # Hinge line, drawn only across the elevon span where it exists.
    for sign in (1, -1):
        ye = np.linspace(sign * ein, sign * eout, 40)
        fig.add_trace(go.Scatter(
            x=ye * 1e3,
            y=np.array([hinge_at(y) for y in ye]) * 1e3,
            mode="lines", line={"color": "red", "width": 2, "dash": "dash"},
            name="Hinge line" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    # Elevon area, shaded so its span and chordwise extent are both obvious.
    for sign in (1, -1):
        ye = np.linspace(sign * ein, sign * eout, 40)
        hinge = np.array([hinge_at(y) for y in ye])
        trail = np.array([te_at(y) for y in ye])
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
            y=np.array([le_at(lo), le_at(hi), te_at(hi), te_at(lo),
                        le_at(lo)]) * 1e3,
            mode="lines", line={"color": "blue", "width": 1, "dash": "dot"},
            fill="toself", fillcolor="rgba(80,120,220,0.15)",
            name="Slipstream" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    # Battery on the centreline and servos outboard, which is the layout that
    # lets both fit: they occupy different spanwise stations rather than
    # competing for the same chord.
    batt_x0 = le_at(0.0) + float(g["battery_station"]) * root_c
    batt_x1 = batt_x0 + am.BATTERY_LENGTH
    fig.add_trace(go.Scatter(
        x=np.array([-0.5 * am.SERVO_WIDTH, 0.5 * am.SERVO_WIDTH,
                    0.5 * am.SERVO_WIDTH, -0.5 * am.SERVO_WIDTH,
                    -0.5 * am.SERVO_WIDTH]) * 1e3,
        y=np.array([batt_x0, batt_x0, batt_x1, batt_x1, batt_x0]) * 1e3,
        mode="lines", line={"color": "green", "width": 2},
        fill="toself", fillcolor="rgba(80,180,80,0.35)",
        name="Battery", hoverinfo="skip",
    ))

    servo_y = float(g["servo_y"])
    servo_c = float(g["servo_chord"])
    station = float(g["servo_chord_frac"])
    for sign in (1, -1):
        y0 = sign * servo_y - 0.5 * am.SERVO_WIDTH
        y1 = sign * servo_y + 0.5 * am.SERVO_WIDTH
        x0 = le_at(servo_y) + station * servo_c - 0.5 * am.SERVO_LENGTH
        x1 = le_at(servo_y) + station * servo_c + 0.5 * am.SERVO_LENGTH
        fig.add_trace(go.Scatter(
            x=np.array([y0, y1, y1, y0, y0]) * 1e3,
            y=np.array([x0, x0, x1, x1, x0]) * 1e3,
            mode="lines", line={"color": "purple", "width": 2},
            fill="toself", fillcolor="rgba(160,80,200,0.4)",
            name="Servo" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    # Wiring reach limits.  Drawn because they are the constraints most likely
    # to be forgotten when reading a planform: nothing about the shape shows
    # that the harness cannot reach further out.
    for limit, colour, label in ((am.MAX_MOTOR_Y, "blue", "Motor reach"),
                                 (am.MAX_SERVO_Y, "purple", "Servo reach")):
        for sign in (1, -1):
            fig.add_trace(go.Scatter(
                x=np.array([sign * limit, sign * limit]) * 1e3,
                y=np.array([le_at(limit), te_at(limit)]) * 1e3,
                mode="lines",
                line={"color": colour, "width": 1, "dash": "longdash"},
                name=label if sign == 1 else None,
                showlegend=sign == 1, hoverinfo="skip",
            ))

    # Props, drawn centred on the leading edge at their spanwise station, which
    # sweeps aft with it.
    theta = np.linspace(0, 2 * np.pi, 60)
    for sign in (1, -1):
        fig.add_trace(go.Scatter(
            x=(sign * motor_y + 0.5 * am.PROP_DIAMETER * np.cos(theta)) * 1e3,
            y=(le_at(motor_y) + 0.5 * am.PROP_DIAMETER * np.sin(theta)) * 1e3,
            mode="lines", line={"color": "blue", "width": 2},
            name="Prop disc" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))

    fig.update_layout(
        title=(f"Planform  --  {float(r['area']) * 1e4:.0f} cm^2,"
               f" AR {float(r['aspect_ratio']):.2f},"
               f" LE sweep {le_sweep:+.0f} deg,"
               f" c/4 {float(r['c4_sweep_deg']):+.1f} deg"),
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
    station = float(g["servo_chord_frac"])
    mac = float(r["mac"])

    # Hinge fraction at *this* section, which is not the design variable when
    # the elevon is constant-chord: x_hinge then sets the elevon width at the
    # mean aerodynamic chord, and the fraction varies along the span.
    x_hinge = float(am.hinge_fraction_at(chord, float(g["x_hinge"]), mac))

    xs, upper, lower, camber = _section_outline(
        chord, thick, float(g["camber_a1"]), float(g["camber_a2"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]) * 1e3,
        y=np.concatenate([upper, lower[::-1]]) * 1e3,
        fill="toself", fillcolor="rgba(200,200,200,0.35)",
        line={"color": "black", "width": 2},
        name="Section", hoverinfo="skip",
    ))

    # The camber line, which is the whole shape story: whether it rises and
    # falls once (ordinary camber, nose-down moment) or turns back up near the
    # trailing edge (reflex, which is what trims a tailless wing).  Drawn
    # against the chord line so the reflex is visible as a crossing rather than
    # having to be inferred from the surfaces.
    fig.add_trace(go.Scatter(
        x=xs * 1e3, y=camber * 1e3,
        mode="lines", line={"color": "black", "width": 1, "dash": "dot"},
        name=f"Camber ({float(r['camber']) * 100:.1f}%)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.array([0.0, chord]) * 1e3, y=np.array([0.0, 0.0]),
        mode="lines", line={"color": "gray", "width": 1},
        name="Chord", hoverinfo="skip",
    ))

    # Battery, drawn at its own depth rather than the section's, so the gap
    # between the box and the surface shows how much room is actually spare.
    # Its forward face is the tight end: that is where the nose runs out of
    # depth, and pushing it further forward is what forces a thicker root.
    batt_x0 = float(g["battery_station"]) * chord
    batt_x1 = batt_x0 + am.BATTERY_LENGTH
    batt_h = 0.5 * am.BATTERY_THICKNESS
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
        x=np.array([sv_x1, servo_chord - float(am.elevon_chord_at(
            servo_chord, float(g["x_hinge"]), mac))]) * 1e3,
        y=np.array([0.0, 0.0]) * 1e3,
        mode="lines", line={"color": "purple", "width": 1, "dash": "dot"},
        name="Pushrod", hoverinfo="skip",
    ))

    # Hinge line through the full local thickness, centred on the camber line
    # rather than on the chord, since that is where the section actually is.
    h_half = float(am.thickness_at(x_hinge, thick)) * 0.5
    h_mid = float(am.camber_line(x_hinge, float(g["camber_a1"]),
                                 float(g["camber_a2"]))) * chord
    fig.add_trace(go.Scatter(
        x=np.array([x_hinge * chord, x_hinge * chord]) * 1e3,
        y=np.array([h_mid - h_half, h_mid + h_half]) * 1e3,
        mode="lines+markers", line={"color": "red", "width": 2, "dash": "dash"},
        marker={"color": "red", "size": 6}, name="Hinge",
    ))

    slack = float(r["volume_slack"]) * 1e3
    fig.update_layout(
        title=(f"Root section  --  {chord * 1e3:.0f} mm chord,"
               f" {float(r['root_tc']) * 100:.1f}% thick,"
               f" {float(r['camber']) * 100:.1f}% camber,"
               f" Cm0 {float(r['cm_c4']):+.3f},"
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
    station = float(g["servo_chord_frac"])

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
    # Hinge fraction at the servo's own section, which is where this plot lives.
    xh_local = float(am.hinge_fraction_at(
        float(g["servo_chord"]), float(g["x_hinge"]), float(am.evaluate(p)["mac"])))
    fig.add_trace(go.Scatter(
        x=[xh_local * 100, xh_local * 100],
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


def authority_figure(p=None, deflection_deg=None):
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

    # Read the deflection back out rather than off the argument, which is None
    # in the usual case: the model derives it from the authority floors and what
    # the linkage can deliver.
    delta = float(r["deflection_deg"])
    derived = "" if deflection_deg is not None else " (derived)"

    fig.update_layout(
        title=f"Control authority at {delta:.2f} deg deflection{derived}",
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


# Units and scaling for the sensitivity table, so a chord in metres and a sweep
# in degrees can be read side by side.  The factor converts the stored value to
# the displayed unit; the gradient is divided by it so it stays "per displayed
# unit" and the numbers are comparable.
_VAR_UNITS = {
    "root_chord": ("mm", 1e3),
    "tip_chord": ("mm", 1e3),
    "root_thickness": ("mm", 1e3),
    "tip_thickness": ("mm", 1e3),
    "x_hinge": ("% chord", 1e2),
    "elevon_inboard_frac": ("% semi", 1e2),
    "motor_frac": ("% semi", 1e2),
    "servo_chord_frac": ("% chord", 1e2),
    "servo_height": ("mm", 1.0),
    "servo_rod_dy": ("mm", 1.0),
    "servo_travel_mm": ("mm", 1.0),
    "flap_x_mm": ("mm", 1.0),
    "flap_y_mm": ("mm", 1.0),
    "servo_span_frac": ("% semi", 1e2),
    "le_sweep_deg": ("deg", 1.0),
    "battery_station": ("% chord", 1e2),
}


def sensitivity_table_html(p=None):
    """Design variables with their sensitivities and what pins them.

    The point of the table is to separate results from artifacts.  A variable
    with a real gradient and no active constraint was genuinely optimized; one
    sitting on a bound was decided by that bound; and one with a vanishing
    gradient was not decided by the objective at all, whatever value it shows.
    """
    p = default_design() if p is None else p
    rows = am.sensitivity(p, opt.BOUNDS)

    # Scale of the largest sensitivity, used to flag the ones small enough that
    # the objective is effectively blind to them.
    biggest = max(abs(r["d_stall"]) for r in rows) or 1.0

    out = [
        "<style>",
        "  .sens { border-collapse: collapse; font-family: system-ui, sans-serif;",
        "          font-size: 13px; margin: 24px auto; max-width: 1100px; }",
        "  .sens th, .sens td { border: 1px solid #ccc; padding: 6px 10px;",
        "                       text-align: right; }",
        "  .sens th { background: #f0f0f0; text-align: center; }",
        "  .sens td.name { text-align: left; font-family: monospace; }",
        "  .sens td.note { text-align: left; color: #555; font-size: 12px; }",
        "  .sens tr.blind { background: #fff4f4; }",
        "  .sens tr.bound { background: #f4f8ff; }",
        "  .sens caption { font-family: system-ui, sans-serif; font-size: 14px;",
        "                  padding: 10px; text-align: left; max-width: 1100px; }",
        "</style>",
        '<table class="sens">',
        "<caption><b>Design variables at the optimum.</b> "
        "d(stall)/dx is what the variable is worth on its own; "
        "d(cost)/dx includes constraint penalties and is what the optimizer "
        "actually followed. A variable is only a genuine result if it has a "
        "real gradient and nothing pinning it &mdash; rows shaded red are "
        "invisible to the objective, rows shaded blue sit on a bound."
        "</caption>",
        "<tr><th>variable</th><th>value</th><th>bounds</th>"
        "<th>d(stall)/dx<br>m/s per unit</th><th>d(cost)/dx</th>"
        "<th>at bound</th><th>pinned by</th></tr>",
    ]

    for r in rows:
        unit, scale = _VAR_UNITS.get(r["name"], ("", 1.0))
        blind = abs(r["d_stall"]) < 1e-6 * biggest
        cls = "blind" if blind else ("bound" if r["at_bound"] or r["pinned_by"]
                                     else "")
        note = ", ".join(r["pinned_by"]) if r["pinned_by"] else ""
        if blind and not note:
            note = "objective is flat in this variable"

        out.append(
            f'<tr class="{cls}">'
            f'<td class="name">{r["name"]}</td>'
            f'<td>{r["value"] * scale:.2f} {unit}</td>'
            f'<td>{r["lower"] * scale:.1f} &ndash; {r["upper"] * scale:.1f}</td>'
            f'<td>{r["d_stall"] / scale:+.3e}</td>'
            f'<td>{r["d_cost"] / scale:+.3e}</td>'
            f'<td>{r["at_bound"] or "&mdash;"}</td>'
            f'<td class="note">{note or "&mdash;"}</td>'
            "</tr>"
        )

    out.append("</table>")

    active = am.active_constraints(p)
    out.append(
        '<p style="font-family: system-ui, sans-serif; font-size: 13px;'
        ' max-width: 1100px; margin: 0 auto 32px;">'
        f'<b>Active constraints:</b> {", ".join(active) if active else "none"}.'
        "</p>"
    )
    return "\n".join(out)


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

    # The table is appended as plain HTML rather than built as a Plotly table,
    # so it can carry its own styling and wrap the explanatory caption.
    html = dashboard(p, label=label).to_html(include_plotlyjs="cdn",
                                             full_html=True)
    table = sensitivity_table_html(p)
    html = html.replace("</body>", f"{table}\n</body>")
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({label})")


if __name__ == "__main__":
    main()
