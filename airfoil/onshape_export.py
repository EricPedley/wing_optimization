"""Fit a cubic B-spline to the analytic root/tip sections and print control
points for Onshape.

The section shape (camber_line + thickness_at, see airfoil_model.py) is
closed-form, not a point table, so there is nothing to hand Onshape directly.
This samples each surface finely, fits a cubic spline through it, and reports
the *fitted spline's* control points -- not the raw samples -- because those
are what a native Onshape sketch spline actually stores.

The intended workflow: build one sketch spline per surface (upper/lower,
root/tip) in Onshape, pin each control point with perpendicular construction
lines back to the origin, dimension those lines, and drive the dimensions
from the numbers printed here. That wiring is manual and per-point, but only
has to be done once -- after that, re-running this script after a design
change gives you a new set of numbers to paste into the variable table
instead of rebuilding the sketch.

Units: Onshape defaults to mm, so everything here is exported in mm even
though airfoil_model works in metres throughout.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, BSpline, make_lsq_spline

from airfoil import airfoil_model as am

M_TO_MM = 1000.0

# Cubic, per the module docstring's convention.
SPLINE_DEGREE = 3

# Control points, per surface. More points track the sharp leading-edge
# curvature and the thin trailing edge better; fewer is less tedious to wire
# up in Onshape. 8 is a reasonable middle ground -- see _fit_spline for how
# it turns into a knot vector.
N_CONTROL_POINTS = 12

# Sample the analytic surface far finer than the spline needs, so the fit is
# limited by the spline's own degrees of freedom rather than by sparse input.
N_SAMPLES = 200


def _knot_vector(n_control):
    """Full clamped knot vector for an n_control-point cubic B-spline.

    Knots are placed on a sqrt spacing rather than uniformly in the chord
    fraction. The section's own leading-edge shape goes as sqrt(x) (see
    thickness_at), so curvature is concentrated in the first few percent of
    chord; uniform knots waste control points on the nearly straight aft
    section and underfit the nose. Squaring a uniform grid pushes knots
    toward x=0, which matches where the curvature actually is.
    """
    n_interior_knots = max(n_control - SPLINE_DEGREE - 1, 0)
    if n_interior_knots > 0:
        interior = np.linspace(0.0, 1.0, n_interior_knots + 2)[1:-1] ** 2
    else:
        interior = np.array([])

    # degree+1 copies of each end, clamping the curve to its endpoints, plus
    # the interior knots in between.
    clamped_ends = np.full(SPLINE_DEGREE + 1, 0.0), np.full(SPLINE_DEGREE + 1, 1.0)
    return np.concatenate([clamped_ends[0], interior, clamped_ends[1]])


def _fit_spline(x, y, n_control):
    """Least-squares cubic B-spline through (x, y), returned as control points.

    splprep's smoothing parameter doesn't let you ask for "N control points"
    directly, so this fixes the interior knot count instead: for a clamped
    cubic B-spline, n_control interior-free control points come from
    n_control - degree - 1 interior knots, which is what splprep's `t`
    argument wants -- see _knot_vector for how they are placed.
    """
    full_knots = _knot_vector(n_control)
    u = np.linspace(0.0, 1.0, len(x))
    tck, _ = splprep([x, y], u=u, k=SPLINE_DEGREE, task=-1, t=full_knots)
    knots, coeffs, degree = tck
    control_points = np.stack(coeffs, axis=-1)
    return control_points, knots


def _fit_spline_y_only(u, y, n_control):
    """Least-squares cubic B-spline Y control points against a fixed u grid.

    Used to fit the lower surface once the upper surface has already fixed
    the X control points: unlike _fit_spline, this never touches X at all, so
    the caller can pair the returned Y coefficients with the upper surface's
    X control points and get a lower-surface spline that shares X exactly --
    which is the point, since Onshape only has to track one set of X
    dimensions per station instead of two.
    """
    full_knots = _knot_vector(n_control)
    spline = make_lsq_spline(u, y, full_knots, k=SPLINE_DEGREE)
    return spline.c, full_knots


def eval_spline(control_points_mm, knots, n=200):
    """Sample a clamped cubic B-spline curve from its control points.

    This is what Onshape actually draws through the control points -- not a
    straight-line connect-the-dots -- so it is what the preview should show
    to faithfully represent what the CAD sketch will look like.
    """
    bx = BSpline(knots, control_points_mm[:, 0], SPLINE_DEGREE)
    by = BSpline(knots, control_points_mm[:, 1], SPLINE_DEGREE)
    uu = np.linspace(0.0, 1.0, n)
    return bx(uu), by(uu)


def section_control_points(chord_m, max_thickness_m, a1, a2,
                           n_control=N_CONTROL_POINTS, n_samples=N_SAMPLES):
    """Fitted cubic-spline control points for one section's two surfaces.

    The lower surface reuses the upper surface's fitted X control points
    rather than fitting its own: CAD only has to dimension one set of X
    values per station this way, with the lower spline's shape carried
    entirely by its Y control points. The two surfaces are close enough in
    curvature (same leading-edge sqrt behaviour, same knot placement) that
    sharing X costs very little fit accuracy -- see the preview to check that
    for any particular section.

    Returns a dict with "upper" and "lower" entries, each a
    (control_points_mm, knots) pair -- control_points_mm is (n_control, 2) in
    the same (x aft, y up) chord-line frame airfoil_model uses, x=0 at the
    leading edge, x=chord at the trailing edge. knots is shared by both
    surfaces since they are fit on the same parameterization.
    """
    x = np.linspace(0.0, 1.0, n_samples)
    half = 0.5 * np.asarray(am.thickness_at(x, max_thickness_m))
    camber = np.asarray(am.camber_line(x, a1, a2)) * chord_m
    x_m = x * chord_m
    u = np.linspace(0.0, 1.0, n_samples)

    upper_cp, knots = _fit_spline(x_m, camber + half, n_control)
    lower_y_cp, _ = _fit_spline_y_only(u, camber - half, n_control)
    lower_cp = np.stack([upper_cp[:, 0], lower_y_cp], axis=-1)

    return {"upper": (upper_cp * M_TO_MM, knots),
            "lower": (lower_cp * M_TO_MM, knots)}


def point_label(i, n_control):
    """Name a control point the way Onshape's spline tool presents it.

    Onshape shows each spline endpoint as an anchor point plus a separate
    tangent-handle point, rather than "control point 0/1". For a clamped
    cubic B-spline those are the same data: the curve passes exactly through
    cp0 and cp[-1], and its tangent there points exactly at cp1 and cp[-2]
    respectively (verified numerically -- the derivative at each end is
    proportional to the vector to the adjacent control point). So cp1 and
    cp[-2] *are* the tangent-handle points; this only renames them so the
    printed list lines up with what Onshape's UI asks you to place.
    """
    if i == 0:
        return "start_anchor"
    if i == 1:
        return "start_tangent_handle"
    if i == n_control - 2:
        return "end_tangent_handle"
    if i == n_control - 1:
        return "end_anchor"
    return f"cp{i}"


def _print_station(name, chord_m, max_thickness_m, a1, a2):
    """Print one station's control points, in the form pasted into Onshape.

    The lower surface's X is now identical to the upper surface's (see
    section_control_points), so its X values are not printed at all --
    reference the matching cad_spline_..._upper_..._x variable already in
    CAD instead of creating a second, redundant one. Y is still independent
    and printed for both surfaces.
    """
    pts = section_control_points(chord_m, max_thickness_m, a1, a2)
    print(f"# {name}: chord={chord_m * M_TO_MM:.3f} mm, "
          f"t_max={max_thickness_m * M_TO_MM:.3f} mm, a1={a1:.4f}, a2={a2:.4f}")
    for surface in ("upper", "lower"):
        print(f"## {name}_{surface}")
        control_points_mm, _ = pts[surface]
        n_control = len(control_points_mm)
        for i, (px, py) in enumerate(control_points_mm):
            label = point_label(i, n_control)
            if surface == "upper":
                print(f"{name}_{surface}_{label}_x = {px:.4f} mm")
            else:
                print(f"# {name}_{surface}_{label}_x = {name}_upper_{label}_x"
                      f"  (shared with upper, {px:.4f} mm)")
            print(f"{name}_{surface}_{label}_y = {py:.4f} mm")
    print()


def _station_geometry(name, chord_m, max_thickness_m, a1, a2):
    return {"name": name, "chord_m": chord_m, "max_thickness_m": max_thickness_m,
            "a1": a1, "a2": a2}


def baseline_stations():
    """Root and tip section parameters from airfoil_model.BASELINE."""
    g = am.unpack(am.BASELINE)
    tip_a1 = float(g["camber_a1"]) + float(
        am.BASELINE[am.DESIGN_VARS.index("tip_camber_da1")])
    tip_a2 = float(g["camber_a2"]) + float(
        am.BASELINE[am.DESIGN_VARS.index("tip_camber_da2")])
    return [
        _station_geometry("root", float(g["root_chord"]),
                          float(g["root_thickness"]),
                          float(g["camber_a1"]), float(g["camber_a2"])),
        _station_geometry("tip", float(g["tip_chord"]),
                          float(g["tip_thickness"]), tip_a1, tip_a2),
    ]


def preview_figure(stations=None, n_control=N_CONTROL_POINTS):
    """Plotly figure comparing the analytic section to the fitted spline.

    One row per station (root, tip). Each row shows the true camber+thickness
    outline as a filled reference, the B-spline curve Onshape would actually
    draw through the fitted control points, and the control points themselves
    with their polygon -- so you can see both the fit error and exactly what
    you'd be dimensioning by hand before committing to the CAD wiring.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    stations = baseline_stations() if stations is None else stations

    fig = make_subplots(rows=len(stations), cols=1,
                        subplot_titles=[s["name"] for s in stations],
                        vertical_spacing=0.12)

    for row, s in enumerate(stations, start=1):
        chord_mm = s["chord_m"] * M_TO_MM
        x = np.linspace(0.0, 1.0, N_SAMPLES)
        half = 0.5 * np.asarray(am.thickness_at(x, s["max_thickness_m"]))
        camber = np.asarray(am.camber_line(x, s["a1"], s["a2"])) * s["chord_m"]
        x_mm = x * chord_mm

        pts = section_control_points(s["chord_m"], s["max_thickness_m"],
                                     s["a1"], s["a2"], n_control=n_control)

        max_err_mm = 0.0
        for surface, sign, color in (("upper", 1.0, "#2a6f97"),
                                     ("lower", -1.0, "#c1440e")):
            y_true_mm = (camber + sign * half) * M_TO_MM
            fig.add_trace(go.Scatter(
                x=x_mm, y=y_true_mm, mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=f"{s['name']} {surface} analytic",
                legendgroup=surface, showlegend=(row == 1)),
                row=row, col=1)

            control_points_mm, knots = pts[surface]
            fit_x, fit_y = eval_spline(control_points_mm, knots)
            fig.add_trace(go.Scatter(
                x=fit_x, y=fit_y, mode="lines",
                line=dict(color=color, width=2),
                name=f"{s['name']} {surface} spline fit",
                legendgroup=surface, showlegend=(row == 1)),
                row=row, col=1)

            n_control_pts = len(control_points_mm)
            labels = [point_label(i, n_control_pts) for i in range(n_control_pts)]
            is_endpoint = np.array(["anchor" in l for l in labels])
            symbols = np.where(is_endpoint, "circle", "circle-open")
            sizes = np.where(is_endpoint, 9, 5)

            fig.add_trace(go.Scatter(
                x=control_points_mm[:, 0], y=control_points_mm[:, 1],
                mode="markers+lines", line=dict(color=color, width=1, dash="dash"),
                marker=dict(color=color, size=sizes, symbol=symbols),
                text=labels, hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f}",
                name=f"{s['name']} {surface} control pts "
                     f"(filled = anchor, open = tangent handle)",
                legendgroup=f"{surface}_cp", showlegend=(row == 1),
                opacity=0.7),
                row=row, col=1)

            y_true_at_fit = np.interp(fit_x, x_mm, y_true_mm)
            max_err_mm = max(max_err_mm, np.max(np.abs(fit_y - y_true_at_fit)))

        fig.update_yaxes(scaleanchor=f"x{row if row > 1 else ''}",
                         scaleratio=1, row=row, col=1,
                         title=f"max fit error: {max_err_mm:.3f} mm")
        fig.update_xaxes(title="chord, mm", row=row, col=1)

    fig.update_layout(
        title=f"Airfoil spline preview ({n_control} control points/surface)",
        height=380 * len(stations), width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def main():
    import argparse
    import io
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=Path, default=None,
                        help="Write an HTML preview to this path instead of "
                             "printing control-point numbers.")
    parser.add_argument("--out", type=Path,
                        default=Path("airfoil/onshape_vars.txt"),
                        help="Also save the printed variable-table listing "
                             "to this file, to keep open while CADding.")
    parser.add_argument("--n-control", type=int, default=N_CONTROL_POINTS)
    args = parser.parse_args()

    if args.preview is not None:
        fig = preview_figure(n_control=args.n_control)
        fig.write_html(str(args.preview))
        print(f"wrote {args.preview}")
        return

    buf = io.StringIO()
    tee = _Tee(sys.stdout, buf)
    real_stdout, sys.stdout = sys.stdout, tee
    try:
        for s in baseline_stations():
            _print_station(s["name"], s["chord_m"], s["max_thickness_m"],
                           s["a1"], s["a2"])
    finally:
        sys.stdout = real_stdout

    args.out.write_text(buf.getvalue())
    print(f"wrote {args.out}")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


if __name__ == "__main__":
    main()
