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
    span = float(g["span"])
    semi = 0.5 * span
    x_hinge = float(g["x_hinge"])
    ein = float(g["elevon_inboard_y"])
    eout = float(g["elevon_outboard_y"])
    motor_y = float(g["motor_y"])

    # x runs aft from the root leading edge, y outboard from the centreline.
    le_sweep = float(g["le_sweep_deg"])

    def chord_at(y):
        # Through the model's own loft, so the constant-chord centre section
        # shows up in the drawing instead of only in the numbers.
        return float(am.local_geometry(span, root_c, tip_c, 0.0, 0.0,
                                       abs(y) / semi)[0])

    def le_at(y):
        return float(am.leading_edge_x(span, abs(y) / semi, le_sweep))

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

    # The mass breakdown behind the CG, read from the model rather than
    # recomputed here so the picture cannot disagree with the constraint.  Shown
    # on hover: five items is too many to label on the plot, but "why is the CG
    # there" is exactly the question this figure gets asked.
    items = am.mass_items(
        span, float(g["root_chord"]), float(g["tip_chord"]),
        float(g["root_thickness"]), float(g["tip_thickness"]),
        le_sweep, float(g["battery_station"]),
        float(g["servo_chord_frac"]), float(g["servo_span_frac"]),
        float(g["motor_standoff"]))
    total = sum(float(m) for m, _ in items.values())
    breakdown = "<br>".join(
        f"{k}: {float(m) * 1e3:.1f} g at {float(s) * 1e3:+.0f} mm"
        f" ({(float(s) - x_cg) * float(m) / total * 1e3:+.1f} mm pull)"
        for k, (m, s) in items.items())

    fig.add_trace(go.Scatter(
        x=[0.0], y=[x_cg * 1e3], mode="markers",
        marker={"color": "black", "size": 13, "symbol": "circle-cross"},
        name=f"CG ({x_cg * 1e3:.0f} mm)",
        hovertemplate=f"CG {x_cg * 1e3:.1f} mm<br>{breakdown}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0.0], y=[x_ac * 1e3], mode="markers",
        marker={"color": "purple", "size": 11, "symbol": "diamond"},
        name=f"AC ({x_ac * 1e3:.0f} mm),  SM {sm * 100:+.1f}%",
        hovertemplate=(f"Aerodynamic centre {x_ac * 1e3:.1f} mm<br>"
                       f"static margin {sm * 100:+.1f}% MAC"
                       f" (floor {am.MIN_STATIC_MARGIN * 100:.0f}%)"
                       "<extra></extra>"),
    ))
    # The static margin itself, as the gap between the two.  Drawn on the
    # centreline because that is where both markers sit.
    fig.add_trace(go.Scatter(
        x=[0.0, 0.0], y=[x_cg * 1e3, x_ac * 1e3],
        mode="lines", line={"color": "purple", "width": 3, "dash": "dot"},
        name=None, showlegend=False, hoverinfo="skip",
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
    #
    # An arc, not a spanwise line.  The limit is a length of wire from the
    # flight controller, so what it bounds is a radius about the FC -- and on a
    # swept wing the outboard stations are further aft, which spends that radius
    # without going any further out.  Drawn as the locus so the picture shows
    # the constraint the model applies rather than the spanwise simplification
    # it used to.
    x_fc = float(am.fc_station(root_c))
    for limit, colour, label in ((am.MAX_MOTOR_WIRE, "blue", "Motor reach"),
                                 (am.MAX_SERVO_WIRE, "purple", "Servo reach")):
        arc = np.linspace(-np.pi / 2, np.pi / 2, 121)
        for sign in (1, -1):
            fig.add_trace(go.Scatter(
                x=sign * limit * np.cos(arc) * 1e3,
                y=(x_fc + limit * np.sin(arc)) * 1e3,
                mode="lines",
                line={"color": colour, "width": 1, "dash": "longdash"},
                name=label if sign == 1 else None,
                showlegend=sign == 1, hoverinfo="skip",
            ))

    # Motors and props, drawn where the model actually puts them, which is
    # *ahead* of the leading edge rather than on it.  The mount pad stands proud
    # by the motor standoff so it is a flat face instead of a knife edge, the
    # body occupies MOTOR_BODY_LENGTH forward of that, and the prop disc sits at
    # its front.  Drawing the disc on the leading edge -- which is what this did
    # before the motor station was modelled -- hides both the standoff and the
    # only mass on the aircraft that sits forward of the wing.
    theta = np.linspace(0, 2 * np.pi, 60)
    standoff = float(g["motor_standoff"])
    prop_y = le_at(motor_y) - standoff - am.MOTOR_BODY_LENGTH
    body_y0 = le_at(motor_y) - standoff
    motor_x = float(am.motor_station(standoff)) + le_at(motor_y)
    for sign in (1, -1):
        fig.add_trace(go.Scatter(
            x=(sign * motor_y + 0.5 * am.PROP_DIAMETER * np.cos(theta)) * 1e3,
            y=(prop_y + 0.5 * am.PROP_DIAMETER * np.sin(theta)) * 1e3,
            mode="lines", line={"color": "blue", "width": 2},
            name="Prop disc" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))
        # Motor body, between the prop plane and the mount face on the wing.
        half_body = 0.5 * am.PROP_DIAMETER * 0.18
        fig.add_trace(go.Scatter(
            x=(sign * motor_y + np.array([-half_body, half_body, half_body,
                                          -half_body, -half_body])) * 1e3,
            y=np.array([prop_y, prop_y, body_y0, body_y0, prop_y]) * 1e3,
            mode="lines", line={"color": "blue", "width": 1},
            fill="toself", fillcolor="rgba(80,80,220,0.25)",
            name="Motor body" if sign == 1 else None,
            showlegend=sign == 1, hoverinfo="skip",
        ))
        # Where the motor's mass acts -- the only item forward of the wing, and
        # so the only one pulling the centre of gravity the right way.
        fig.add_trace(go.Scatter(
            x=[sign * motor_y * 1e3], y=[motor_x * 1e3],
            mode="markers",
            marker={"color": "blue", "size": 8, "symbol": "x-thin",
                    "line": {"color": "blue", "width": 2}},
            name=(f"Motor mass ({am.MOTOR_MASS * 1e3:.1f} g, "
                  f"{float(am.motor_station(standoff)) * 1e3:+.0f} mm)")
            if sign == 1 else None,
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


def _box(x0, x1, depth, chord, thickness, a1, a2):
    """A rigid box seated in the section, as plotly x/y in millimetres.

    Boxes used to be drawn centred on the chord line, which was right while the
    section was symmetric and wrong the moment it was not: a cambered section
    sits above its chord, so a box centred on y=0 hangs out through the lower
    surface even when the model says it fits.  That looked like a packaging
    failure the model was ignoring, when in fact the model was right and only
    the drawing was wrong.

    Seated against the *highest* point of the lower surface over its own
    footprint, which is where a real box lands: it rests on whichever part of
    the inner skin rises furthest into it.  The remaining gap to the upper
    surface is then the real spare room, which is what the drawing is for.
    """
    xs = np.linspace(x0 / chord, x1 / chord, 9)
    half = 0.5 * np.asarray(am.thickness_at(xs, thickness))
    camber = np.asarray(am.camber_line(xs, a1, a2)) * chord
    floor = float(np.max(camber - half))
    return {
        "x": np.array([x0, x1, x1, x0, x0]) * 1e3,
        "y": np.array([floor, floor, floor + depth, floor + depth, floor]) * 1e3,
    }


def section_figure(p=None):
    """Root section with the battery and servo drawn to scale inside it.

    The tip section is overlaid on the same axes, drawn to its own chord, so the
    two shapes can be compared directly.  They are no longer the same section
    scaled: the tip carries its own camber, and the difference between them is
    the washout that decides which end of the wing stalls first.
    """
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

    # Tip section, at its own chord and its own camber.  Outline only, no fill,
    # so it reads as an overlay rather than competing with the root for
    # attention -- the root is the one with the packaging problem, the tip is
    # here to be compared against it.
    tip_chord = float(g["tip_chord"])
    tip_xs, tip_upper, tip_lower, tip_camber = _section_outline(
        tip_chord, float(g["tip_thickness"]),
        float(g["tip_camber_a1"]), float(g["tip_camber_a2"]))
    fig.add_trace(go.Scatter(
        x=np.concatenate([tip_xs, tip_xs[::-1]]) * 1e3,
        y=np.concatenate([tip_upper, tip_lower[::-1]]) * 1e3,
        mode="lines", line={"color": "steelblue", "width": 2},
        name=f"Tip section ({float(r['tip_camber']) * 100:.1f}% camber)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=tip_xs * 1e3, y=tip_camber * 1e3,
        mode="lines", line={"color": "steelblue", "width": 1, "dash": "dot"},
        name="Tip camber", hoverinfo="skip", showlegend=False,
    ))

    # Battery, drawn at its own depth rather than the section's, so the gap
    # between the box and the surface shows how much room is actually spare.
    # Its forward face is the tight end: that is where the nose runs out of
    # depth, and pushing it further forward is what forces a thicker root.
    batt_x0 = float(g["battery_station"]) * chord
    batt_x1 = batt_x0 + am.BATTERY_LENGTH
    fig.add_trace(go.Scatter(
        **_box(batt_x0, batt_x1, am.BATTERY_THICKNESS, chord, thick,
               float(g["camber_a1"]), float(g["camber_a2"])),
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
    # Seated in its *own* section, not the root's: the servo sits outboard where
    # the wing is shorter, thinner, and -- now that the tip carries its own
    # camber -- a different shape.  Drawing it against the root section would
    # show it fitting in room that does not exist where it actually lives.
    sv_a1, sv_a2 = am.local_camber(
        float(g["span"]),
        float(g["camber_a1"]), float(g["camber_a2"]),
        float(g["tip_camber_a1"]), float(g["tip_camber_a2"]),
        float(g["servo_span_frac"]))
    servo_box = _box(sv_x0, sv_x1, am.SERVO_DEPTH, servo_chord,
                     float(g["servo_thickness"]), float(sv_a1), float(sv_a2))
    fig.add_trace(go.Scatter(
        **servo_box,
        mode="lines", line={"color": "purple", "width": 2, "dash": "dash"},
        fill="toself", fillcolor="rgba(160,80,200,0.25)",
        name=f"Servo (at {float(g['servo_span_frac']) * 100:.0f}% semi-span)",
        hoverinfo="skip",
    ))

    # Pushrod from the servo output to the hinge, straight-line schematic.  Run
    # at the servo's own mid-height rather than along the chord line, so it
    # leaves the servo where the servo actually is.
    sv_mid = 0.5 * (servo_box["y"][0] + servo_box["y"][2])
    fig.add_trace(go.Scatter(
        x=np.array([sv_x1 * 1e3, (servo_chord - float(am.elevon_chord_at(
            servo_chord, float(g["x_hinge"]), mac))) * 1e3]),
        y=np.array([sv_mid, sv_mid]),
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
        title=(f"Sections  --  root {chord * 1e3:.0f} mm /"
               f" tip {tip_chord * 1e3:.0f} mm,"
               f" camber {float(r['camber']) * 100:.1f}% /"
               f" {float(r['tip_camber']) * 100:.1f}%"
               f" (washout {float(r['washout']) * 100:.1f}%),"
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
    "span": ("mm", 1e3),
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
    "motor_standoff_mm": ("mm", 1.0),
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
