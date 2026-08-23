"""Build the half-planform sketch in an Onshape part studio from the variables.

The wing is symmetric, so only the starboard half is drawn; the solid gets
mirrored in CAD once it exists.  That leaves one servo pocket, one motor mount
and one hinge line to detail instead of two, and makes it impossible for the
halves to disagree -- a mirror cannot drift, two hand-built halves can.  The
centre section is the exception: the battery straddles the centreline, so its
pocket belongs *after* the mirror rather than in this sketch.

Axes follow the robotics convention the airframe is flown in: +x forward, +y out
the starboard side, +z up.  Chord therefore runs aft along -x with the root
leading edge at the origin, which is why every chordwise quantity is negated
here.  The variable studio keeps the model's own convention -- "aft of the root
leading edge, positive" -- because that is what the aerodynamic code means by
those numbers, and the sketch is the right place to adapt between the two.

Geometry is *computed* from the variables rather than constrained by dimensions.
A FeatureScript sketch built this way cannot be dragged in the Onshape UI: to
move it you change the design vector and re-push.  That is the intended
direction of authority -- the optimizer decides the geometry, and a human
dragging a line in CAD is exactly the drift the variable studio exists to
prevent.

The whole sketch goes up as one FeatureScript body in a single request.  The API
is rate limited, so the working loop is deliberately two calls: evaluate the
same expressions to check them against the model, then write once.

    uv run python -m airfoil.onshape_sketch --check   # verify, no write
    uv run python -m airfoil.onshape_sketch           # write the sketch
"""

import argparse
import json

import airfoil.airfoil_model as am
import airfoil.onshape as osh

# The part studio the parametric rebuild lives in.  Distinct from the Airframe
# studio, which holds the hand-built v0 and is left alone.
PART_STUDIO_ID = "27cacff903171ab91eef492e"

# Sketch feature name, used to find and replace the previous one on a re-run so
# repeated pushes do not stack duplicate sketches.
SKETCH_NAME = "Planform half (generated)"

# The Feature Studio holding the generated custom feature, and the feature
# itself.  A custom feature cannot be defined inline in a part studio: it lives
# in a Feature Studio, is published as a version, and is referenced from the
# part studio by namespace.
FEATURE_STUDIO_NAME = "Generated features"
FEATURE_FUNCTION = "generatedPlanform"
FEATURE_NAME = "Planform half (generated)"

# FeatureScript language version the generated source declares.  Pinned rather
# than floating so a change in Onshape's current version cannot silently alter
# the meaning of the emitted code.  Onshape stamps a newly created Feature
# Studio with the version current at the time, which is where this came from --
# raising it is a deliberate act, not something to inherit by accident.
FS_VERSION = "3044"

# Onshape's default Top plane is the xy plane: +x forward, +y to starboard in
# this frame.  Referenced by its stable creation id rather than by picking a
# face, so the sketch does not depend on anything already drawn.
SKETCH_PLANE = 'qCreatedBy(makeId("Top"), EntityType.FACE)'


# --- The geometry, as FeatureScript expressions --------------------------------
#
# Each point is a pair of expression strings over the variable studio.  They are
# built once and used twice -- to compute the points for verification, and to
# emit the sketch -- so the thing that is checked is literally the thing that is
# drawn, rather than a reimplementation of it that could drift.


def var(name):
    """A variable-studio lookup, as FeatureScript source.

    Written as ``getVariable`` rather than the ``#name`` shorthand: the ``#``
    form is Onshape's *expression* syntax, understood in feature dialog fields
    and in variable expressions, but it is not FeatureScript source and does not
    parse inside a function body.
    """
    return f'getVariable(context, "{name}")'


def _aft(expr):
    """Chordwise expression, flipped from 'aft is positive' onto the -x axis."""
    return f"-({expr})"


def _le_offset_at(y_expr):
    """Leading-edge offset aft of the root LE at a spanwise station.

    Piecewise, not linear.  The centre section is a straight extrusion of the
    root -- unswept as well as untapered -- so the leading edge stays at zero
    out to ``cad_root_section_y`` and only then rakes aft.  Interpolating from
    the tip instead looks right at both ends and is wrong everywhere between,
    which is exactly the error the CAD-versus-model check caught: it grew
    steadily inboard, reaching a millimetre at the inboard hinge.

    ``max(y - y0, 0)`` rather than a conditional so the expression stays a
    single algebraic form, valid on both sides of the breakpoint.
    """
    y0 = var("cad_root_section_y")
    swept = f"max(({y_expr}) - {y0}, 0 * millimeter)"
    outer = f"({var('cad_semi_span')} - {y0})"
    return f"({swept} / {outer}) * {var('cad_tip_le_offset')}"


def geometry_points():
    """Every named point in the sketch, as (name, x_expr, y_expr).

    The outline is walked leading edge outboard, then trailing edge back inboard,
    so the four corners close into a loop in order.

    The hinge ends sit at the elevon's own spanwise stations, one elevon chord
    forward of the local trailing edge.  The trailing edge is reconstructed from
    the sweep and the local chord rather than assumed, which is what keeps the
    hinge on it as the planform changes -- and, because the elevon is
    constant-chord, what makes the hinge line parallel to the trailing edge.
    """
    e_in = var("cad_elevon_inboard_y")
    e_out = var("cad_elevon_outboard_y")
    elevon_chord = var("cad_elevon_chord")

    inboard_te = (f"{_le_offset_at(e_in)}"
                  f" + {var('cad_wing_chord_at_elevon_inboard')}")
    outboard_te = (f"{_le_offset_at(e_out)}"
                   f" + {var('cad_wing_chord_at_elevon_outboard')}")

    zero = "0 * millimeter"
    y0 = var("cad_root_section_y")
    root_chord = var("cad_root_chord")

    # Six corners, not four.  The constant-chord centre section means the
    # leading and trailing edges both have a break at ``cad_root_section_y``:
    # parallel-sided inboard of it, tapering outboard.  A four-point outline
    # cuts straight from the root to the tip and loses the centre strip
    # entirely.
    return [
        # Leading edge, centreline outboard.
        ("rootLE", zero, zero),
        ("breakLE", zero, y0),
        ("tipLE", _aft(var("cad_tip_le_offset")), var("cad_semi_span")),
        # Trailing edge, tip back inboard.
        ("tipTE",
         _aft(f"{var('cad_tip_le_offset')} + {var('cad_tip_chord')}"),
         var("cad_semi_span")),
        ("breakTE", _aft(root_chord), y0),
        ("rootTE", _aft(root_chord), zero),
        # Hinge line ends.
        ("hingeInboard", _aft(f"{inboard_te} - {elevon_chord}"), e_in),
        ("hingeOutboard", _aft(f"{outboard_te} - {elevon_chord}"), e_out),
    ]


# Segments to draw, as pairs of point names.  The outline closes back on rootLE;
# the hinge is a separate open segment because it is a split line for a later
# feature, not a boundary of the planform.
OUTLINE_LOOP = ["rootLE", "breakLE", "tipLE", "tipTE", "breakTE", "rootTE"]
EXTRA_SEGMENTS = [("hingeInboard", "hingeOutboard")]


def _segments():
    """Every line segment, as (id, start_name, end_name)."""
    pairs = [(OUTLINE_LOOP[i], OUTLINE_LOOP[(i + 1) % len(OUTLINE_LOOP)])
             for i in range(len(OUTLINE_LOOP))]
    pairs += EXTRA_SEGMENTS
    return [(f"seg{i}", a, b) for i, (a, b) in enumerate(pairs)]


# --- FeatureScript emission ---------------------------------------------------


def _point_definitions():
    """FeatureScript `var` lines defining every point as a 2D vector.

    Emitted as named variables rather than inlined into each segment so that a
    shared corner is computed once and both segments meeting there use the same
    value -- no chance of two nearly-equal numbers leaving a hairline gap that
    breaks the closed loop.
    """
    lines = []
    for name, x_expr, y_expr in geometry_points():
        lines.append(f"        var {name} = vector({x_expr}, {y_expr});")
    return "\n".join(lines)


def sketch_body():
    """The FeatureScript that creates the sketch, without a feature wrapper.

    Kept separate from the feature envelope so the identical body can be run
    through the evaluation endpoint, where it computes and returns the points
    instead of drawing them.
    """
    segments = "\n".join(
        f'        skLineSegment(sketch, "{seg_id}",'
        f' {{ "start" : {a}, "end" : {b} }});'
        for seg_id, a, b in _segments())

    return f"""
{_point_definitions()}

        var sketch = newSketch(context, id + "planformHalf", {{
                "sketchPlane" : {SKETCH_PLANE}
        }});

{segments}

        skSolve(sketch);
"""


def verification_script():
    """FeatureScript that returns every point instead of drawing it.

    The same expressions as :func:`sketch_body`, evaluated and handed back in
    millimetres so they can be checked against the model before anything is
    written.  This is the cheap half of the two-call loop: it proves the
    geometry without spending a write on it.
    """
    reported = ",\n".join(
        f'            "{name}_x" : {name}[0] / millimeter,\n'
        f'            "{name}_y" : {name}[1] / millimeter'
        for name, _, _ in geometry_points())

    return f"""
    function(context is Context, queries) {{
{_point_definitions()}

        return {{
{reported}
        }};
    }}
    """


def feature_studio_source():
    """The complete Feature Studio source defining the planform feature.

    Onshape has no inline-FeatureScript feature: a custom feature has to live in
    a Feature Studio, be published as a version, and then be referenced from the
    part studio by namespace.  This is that source.

    The feature takes no parameters -- everything it needs comes from the
    variable studio -- so the precondition is empty and the dialog is bare.  That
    is deliberate: a parameter here would be a number a human could set in CAD,
    which is precisely the authority this pipeline keeps in the design vector.
    """
    return f"""FeatureScript {FS_VERSION};
import(path : "onshape/std/geometry.fs", version : "{FS_VERSION}.0");

/**
 * Half-planform of the tailsitter wing, generated from the variable studio.
 *
 * Starboard half only; the solid is mirrored downstream.  Axes are the
 * robotics frame the aircraft is flown in: +x forward, +y to starboard, so the
 * chord runs aft along -x from the root leading edge at the origin.
 *
 * Every coordinate is read from the variable studio rather than written here,
 * so re-pushing a new design vector moves this geometry.  Do not edit the
 * numbers in CAD -- edit the design vector and re-push.
 */
annotation {{ "Feature Type Name" : "{FEATURE_NAME}" }}
export const {FEATURE_FUNCTION} = defineFeature(
    function(context is Context, id is Id, definition is map)
    precondition
    {{
    }}
    {{
{sketch_body()}
    }});
"""


# --- Expected geometry, straight from the model -------------------------------


def expected_points(p=None):
    """Where each point should land, computed in Python from the design vector.

    Deliberately a second, independent path to the same numbers: the CAD side
    reaches them through the variable studio and FeatureScript, this side
    through the model directly.  Agreement means the whole chain -- optimizer,
    variable push, sketch expressions -- is consistent.  Units are millimetres,
    signs are the robotics frame.
    """
    if p is None:
        cached = __import__("airfoil.optimize", fromlist=["x"]).load_optimum()
        p = am.BASELINE if cached is None else cached

    g = am.unpack(p)
    r = am.evaluate(p)
    span = float(g["span"])
    semi = 0.5 * span
    mm = 1e3

    root_chord = float(g["root_chord"])
    tip_chord = float(g["tip_chord"])
    tip_le = float(am.leading_edge_x(span, 1.0, float(g["le_sweep_deg"])))
    mac = float(r["mac"])
    elevon_chord = float(am.elevon_chord_at(mac, float(g["x_hinge"]), mac))

    def le_at(y):
        return float(am.leading_edge_x(span, y / semi,
                                       float(g["le_sweep_deg"])))

    def chord_at(y):
        """Local chord at a spanwise station, in metres.

        Routed through the model's own ``local_geometry`` rather than
        interpolated here.  Interpolating looked equivalent and was not: the
        elevon stations are metres, ``local_geometry`` takes a span fraction,
        and mixing the two put the inboard hinge 3 mm off -- which the
        comparison against CAD is exactly what caught.
        """
        return float(am.local_geometry(
            span, root_chord, tip_chord,
            float(g["root_thickness"]), float(g["tip_thickness"]),
            y / semi)[0])

    e_in = float(g["elevon_inboard_y"])
    e_out = float(g["elevon_outboard_y"])

    # Outboard edge of the constant-chord centre section, where the leading
    # edge starts to rake aft and the chord starts to taper.
    y0 = float(am.root_section_fraction(span)) * semi

    return {
        "rootLE_x": 0.0, "rootLE_y": 0.0,
        "breakLE_x": 0.0, "breakLE_y": y0 * mm,
        "tipLE_x": -tip_le * mm, "tipLE_y": semi * mm,
        "tipTE_x": -(tip_le + tip_chord) * mm, "tipTE_y": semi * mm,
        "breakTE_x": -root_chord * mm, "breakTE_y": y0 * mm,
        "rootTE_x": -root_chord * mm, "rootTE_y": 0.0,
        "hingeInboard_x": -(le_at(e_in) + chord_at(e_in) - elevon_chord) * mm,
        "hingeInboard_y": e_in * mm,
        "hingeOutboard_x": -(le_at(e_out) + chord_at(e_out) - elevon_chord) * mm,
        "hingeOutboard_y": e_out * mm,
    }


def check(p=None, part_studio_id=PART_STUDIO_ID, tolerance=1e-3):
    """Evaluate the sketch expressions in CAD and compare against the model.

    One API call.  Returns (results, problems), where results maps each point
    coordinate to (model value, CAD value) in millimetres.
    """
    response = osh._request(
        "POST",
        f"/partstudios/d/{osh.DOCUMENT_ID}/w/{osh.WORKSPACE_ID}"
        f"/e/{part_studio_id}/featurescript",
        body={"script": verification_script(), "queries": {}})

    entries = response["result"]["value"]
    got = {e["key"]["value"]: e["value"]["value"] for e in entries}
    want = expected_points(p)

    results = {}
    problems = []
    for name, value in sorted(want.items()):
        cad = got.get(name)
        if cad is None:
            problems.append(f"{name}: not returned by FeatureScript")
            continue
        results[name] = (value, cad)
        if abs(cad - value) > tolerance:
            problems.append(
                f"{name}: model {value:.4f} mm, CAD {cad:.4f} mm"
                f" (off by {abs(cad - value):.4f})")

    return results, problems


# --- Writing ------------------------------------------------------------------


def existing_sketch_id(part_studio_id=PART_STUDIO_ID):
    """Feature id of a previously generated sketch, if one is there.

    Matched by name because Onshape assigns its own feature ids on creation, so
    the id chosen in the request is not the one that comes back.
    """
    response = osh._request(
        "GET",
        f"/partstudios/d/{osh.DOCUMENT_ID}/w/{osh.WORKSPACE_ID}"
        f"/e/{part_studio_id}/features")
    for feature in response.get("features", []):
        # Matched on the feature *type* -- the exported function name -- rather
        # than on the display name, which a user is free to rename in the UI
        # without changing what the feature is.
        if feature.get("featureType") == FEATURE_FUNCTION:
            return feature.get("featureId")
    return None


def find_feature_studio():
    """Element id of the generated Feature Studio, or None if it is not there."""
    for element in osh.elements():
        if (element.get("elementType") == "FEATURESTUDIO"
                and element.get("name") == FEATURE_STUDIO_NAME):
            return element["id"]
    return None


def create_feature_studio():
    """Create the Feature Studio tab that will hold the generated feature."""
    response = osh._request(
        "POST",
        f"/featurestudios/d/{osh.DOCUMENT_ID}/w/{osh.WORKSPACE_ID}",
        body={"name": FEATURE_STUDIO_NAME})
    return response["id"]


def write_feature_studio(feature_studio_id, source=None):
    """Replace the Feature Studio's contents with the generated source."""
    return osh._request(
        "POST",
        f"/featurestudios/d/{osh.DOCUMENT_ID}/w/{osh.WORKSPACE_ID}"
        f"/e/{feature_studio_id}",
        body={"contents": source or feature_studio_source()})


def create_version(name):
    """Publish a version of the document.

    A part studio can only reference a custom feature from a *version*, not from
    the live workspace, so every change to the generated source needs a new one.
    That is the main ongoing cost of the custom-feature route.
    """
    response = osh._request(
        "POST",
        "/documents/d/%s/versions" % osh.DOCUMENT_ID,
        body={"name": name, "documentId": osh.DOCUMENT_ID,
              "workspaceId": osh.WORKSPACE_ID})
    return response["id"]


def feature_studio_microversion(feature_studio_id, version_id):
    """Microversion of the Feature Studio as of a published version.

    The namespace identifies the code by *microversion*, not by version id: a
    version is a document-wide label, while the microversion pins the exact
    state of this one element.  Reading it back from the version rather than
    from the workspace is what makes the reference stable -- the workspace
    microversion moves on the next edit, and a namespace pointing at it would
    silently start resolving to different code.
    """
    response = osh._request(
        "GET",
        f"/featurestudios/d/{osh.DOCUMENT_ID}/v/{version_id}"
        f"/e/{feature_studio_id}")
    return response["sourceMicroversion"]


def add_feature(feature_studio_id, version_id, part_studio_id=PART_STUDIO_ID):
    """Add the custom feature to the part studio, referencing the version.

    ``featureType`` is the exported function's name, and ``namespace`` is what
    tells Onshape where to find it.  Within one document the reference is local
    -- element and microversion, with no document id -- which is why this is
    ``e...::m...`` rather than the document/version/element triple the phrase
    "the document id, version id, and element id" suggests.  A built-in feature
    has no namespace at all; that field is the only structural difference.
    """
    microversion = feature_studio_microversion(feature_studio_id, version_id)
    body = {
        "feature": {
            "btType": "BTMFeature-134",
            "featureType": FEATURE_FUNCTION,
            "name": FEATURE_NAME,
            "namespace": f"e{feature_studio_id}::m{microversion}",
            "suppressed": False,
            "parameters": [],
        },
    }
    return osh._request(
        "POST",
        f"/partstudios/d/{osh.DOCUMENT_ID}/w/{osh.WORKSPACE_ID}"
        f"/e/{part_studio_id}/features",
        body=body)


def delete_feature(feature_id, part_studio_id=PART_STUDIO_ID):
    """Remove one feature from the part studio by id."""
    return osh._request(
        "DELETE",
        f"/partstudios/d/{osh.DOCUMENT_ID}/w/{osh.WORKSPACE_ID}"
        f"/e/{part_studio_id}/features/featureid/{feature_id}")


def create_sketch(part_studio_id=PART_STUDIO_ID):
    """Publish the generated feature and add it to the part studio.

    Four calls: ensure the Feature Studio exists, write the source, version it,
    then reference it.  Reported step by step because the API is rate limited
    and a failure partway through leaves the document in a state worth knowing
    about -- a written but unversioned studio is harmless, a version is
    permanent.
    """
    studio = find_feature_studio()
    if studio is None:
        studio = create_feature_studio()
        print(f"  created Feature Studio {FEATURE_STUDIO_NAME!r}")
    else:
        print(f"  reusing Feature Studio {FEATURE_STUDIO_NAME!r}")

    write_feature_studio(studio)
    print("  wrote the generated source")

    version = create_version(f"{FEATURE_NAME} source")
    print(f"  published version {version}")

    # Replace the previous instance rather than stacking a second one.  A
    # re-run is meant to update the geometry, and Onshape is perfectly happy to
    # hold two copies of the same generated feature otherwise.
    existing = existing_sketch_id(part_studio_id)
    if existing is not None:
        delete_feature(existing, part_studio_id)
        print("  removed the previous feature")

    add_feature(studio, version, part_studio_id)
    print("  added the feature to the part studio")
    return studio, version


# --- CLI ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="evaluate the sketch points against the model "
                             "without writing anything")
    parser.add_argument("--show", action="store_true",
                        help="print the FeatureScript that would be sent")
    args = parser.parse_args()

    if args.show:
        print(feature_script())
        return

    results, problems = check()
    width = max(len(n) for n in results) if results else 10
    for name, (want, got) in results.items():
        flag = "" if abs(got - want) < 1e-3 else "   <-- MISMATCH"
        print(f"  {name:<{width}}  model {want:>10.4f}   CAD {got:>10.4f}{flag}")

    if problems:
        print(f"\n{len(problems)} problems:")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("\nGeometry verified against the model.")

    if args.check:
        return

    create_sketch()
    print(f"Wrote {SKETCH_NAME!r}.")


if __name__ == "__main__":
    main()
