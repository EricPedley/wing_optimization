"""Push the design point into an Onshape variable studio over the REST API.

The optimizer produces a design vector; CAD needs dimensions.  Retyping the
numbers is where a model and its drawing drift apart, so this module makes the
variable studio a *derived* artifact: it is overwritten from the design vector
and never edited by hand.  If a sketch references only these variables, the CAD
cannot silently disagree with the model.

Two sets of variables are written, because they answer different questions.
The raw design vector is what the optimizer actually decided, in its own units,
and is what you compare against ``optimum.json`` when something looks wrong.
The derived set is what a sketch can consume directly -- positions in mm from
the centreline and the root leading edge, angles in degrees -- so that no
arithmetic happens inside Onshape where it would be invisible to this model.
Derived names are prefixed to keep the two apart at a glance.

Onshape's variable API replaces the *whole* studio on each POST: variables not
present in the payload are deleted.  That is the behaviour we want -- a stale
variable left behind from an earlier schema is exactly the kind of thing that
would keep a sketch alive while the model has moved on -- but it means this must
never be pointed at a studio that also holds hand-authored variables.

Authentication uses an Onshape API key pair as HTTP Basic credentials, read from
the environment or from ``~/.onshape_keys.json``, which is outside the repository
so that no gitignore rule stands between the secret and a commit.  Onshape also
offers a signed-request scheme; Basic over TLS is supported for API keys and
avoids a signing implementation whose failures are hard to distinguish from bad
credentials.

    export ONSHAPE_ACCESS_KEY=...      # from https://dev-portal.onshape.com
    export ONSHAPE_SECRET_KEY=...

    uv run python -m airfoil.onshape --list     # find the studio's element id
    uv run python -m airfoil.onshape --dry-run  # print what would be written
    uv run python -m airfoil.onshape            # write it
"""

import argparse
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import airfoil.airfoil_model as am
import airfoil.optimize as opt

BASE_URL = "https://cad.onshape.com"

# Onshape versions its API in the path.  Pinned rather than floating: a bump can
# change response shapes, and this should fail loudly against a known version
# instead of quietly reading a field that moved.
API_VERSION = "v10"

# The document this design lives in.  Workspace rather than version, because the
# point is to drive the live model.
DOCUMENT_ID = "fc16e035b9272e1f16de9511"
WORKSPACE_ID = "8283d374293017844d6ea3cc"

# The variable studio tab.  Pinned so the common path is one request rather than
# two; ``--list`` re-resolves it by name if the tab is ever renamed or recreated,
# which changes the id.
VARIABLE_STUDIO_NAME = "Variable Studio 1"
VARIABLE_STUDIO_ID = "582f9860e2a4d3ebb75947a9"

# A part studio that inherits the variable studio, used only to check that the
# variables evaluate there -- which is the thing a sketch depends on and the one
# thing the variables endpoint cannot report.
AIRFRAME_STUDIO_ID = "95f58a3d91f86a3c91fa172e"

# Credentials file, used when the environment does not carry the keys.  JSON with
# "access_key" and "secret_key".  Deliberately outside the repository: a secret
# inside the working tree is one `git add -A` away from being committed, and the
# gitignore that would prevent that is itself a file someone can change.
CREDENTIALS_PATH = Path.home() / ".onshape_keys.json"


# --- Authentication -----------------------------------------------------------


def credentials():
    """Onshape API key pair, from the environment or the credentials file.

    The environment wins so a shell can override a stored key without editing
    anything, which is what you want when testing against a second document.
    """
    access = os.environ.get("ONSHAPE_ACCESS_KEY")
    secret = os.environ.get("ONSHAPE_SECRET_KEY")
    if access and secret:
        return access, secret

    if CREDENTIALS_PATH.exists():
        data = json.loads(CREDENTIALS_PATH.read_text())
        access = data.get("access_key")
        secret = data.get("secret_key")
        if access and secret:
            return access, secret

    raise RuntimeError(
        "No Onshape credentials.  Create an API key at "
        "https://dev-portal.onshape.com (scopes: read and write), then either\n"
        "  export ONSHAPE_ACCESS_KEY=... ONSHAPE_SECRET_KEY=...\n"
        f"or write {{\"access_key\": ..., \"secret_key\": ...}} to {CREDENTIALS_PATH}"
    )


def _request(method, path, body=None, query=None):
    """One authenticated API call, returning parsed JSON.

    urllib rather than requests: the whole client is three endpoints, and a
    dependency that exists only to save a dozen lines is a dependency that has
    to be kept current for no return.
    """
    access, secret = credentials()
    url = f"{BASE_URL}/api/{API_VERSION}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    data = None if body is None else json.dumps(body).encode()
    token = base64.b64encode(f"{access}:{secret}".encode()).decode()
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Basic {token}",
        "Accept": "application/json;charset=UTF-8;qs=0.09",
        "Content-Type": "application/json",
    })

    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"Onshape {method} {path} failed: {exc.code} {exc.reason}\n{detail}"
        ) from exc

    return json.loads(payload) if payload else None


# --- Document introspection ---------------------------------------------------


def elements(document_id=DOCUMENT_ID, workspace_id=WORKSPACE_ID):
    """Every tab in the workspace, as returned by the API."""
    return _request(
        "GET", f"/documents/d/{document_id}/w/{workspace_id}/elements")


def find_variable_studio(name=VARIABLE_STUDIO_NAME, document_id=DOCUMENT_ID,
                         workspace_id=WORKSPACE_ID):
    """Element id of the variable studio with this tab name.

    Resolving by name rather than trusting the id pasted from a browser URL,
    which points at whichever tab happened to be open.  Raises with the full tab
    list rather than returning None, because a silent miss here would otherwise
    surface as a confusing permissions error from the variables endpoint.
    """
    found = elements(document_id, workspace_id)
    for element in found:
        if (element.get("elementType") == "VARIABLESTUDIO"
                and element.get("name") == name):
            return element["id"]

    listing = "\n".join(
        f"  {e.get('elementType', '?'):<16} {e.get('name', '?')}  ({e.get('id')})"
        for e in found)
    raise RuntimeError(
        f"No variable studio named {name!r} in this workspace.  Tabs:\n{listing}")


def studio_id():
    """The configured studio id, resolved by name if it has not been pinned."""
    return VARIABLE_STUDIO_ID or find_variable_studio()


# --- Design vector to Onshape variables ---------------------------------------
#
# Onshape variables carry a type and a value string with units.  LENGTH and
# ANGLE values must state their unit; NUMBER values are dimensionless and must
# not.  Getting this wrong is accepted by the API and only fails later inside a
# sketch, so the unit is attached here where it can be read next to the quantity
# it belongs to.

LENGTH = "LENGTH"
ANGLE = "ANGLE"
NUMBER = "NUMBER"


def _variable(name, var_type, value, description):
    """One Onshape variable.

    The literal goes in ``expression``, not ``value``.  Onshape treats a
    variable's expression as its definition and ``value`` as the read-only
    result of evaluating it, so a POST that sets ``value`` is accepted with a
    200 and stores a variable with an empty expression -- named, typed, and
    completely useless in a sketch.  There is no error to catch, which is why
    the round-trip check in ``verify`` exists.
    """
    if var_type == LENGTH:
        expression = f"{value * 1e3:.4f} mm"
    elif var_type == ANGLE:
        expression = f"{value:.4f} deg"
    else:
        expression = f"{value:.6f}"
    return {"type": var_type, "name": name, "expression": expression,
            "description": description}


def design_variables(p):
    """The raw design vector, one Onshape variable per element.

    Named exactly as in :data:`airfoil.airfoil_model.DESIGN_VARS` so a value in
    CAD can be traced back to a column of ``optimum.json`` without a lookup
    table.  Fractions stay fractions here rather than being converted, because
    this set exists to be comparable with the model, not to be drawn with.
    """
    values = [float(v) for v in p]
    types = {
        "root_chord": LENGTH,
        "tip_chord": LENGTH,
        "root_thickness": LENGTH,
        "tip_thickness": LENGTH,
        "x_hinge": NUMBER,
        "elevon_inboard_frac": NUMBER,
        "motor_frac": NUMBER,
        "servo_station": NUMBER,
        "servo_span_frac": NUMBER,
        "le_sweep_deg": ANGLE,
        "battery_station": NUMBER,
    }
    notes = {
        "root_chord": "Root chord",
        "tip_chord": "Tip chord",
        "root_thickness": "Maximum section thickness at the root",
        "tip_thickness": "Maximum section thickness at the tip",
        "x_hinge": "Hinge station as a chord fraction at the MAC",
        "elevon_inboard_frac": "Elevon inboard end, fraction of semi-span",
        "motor_frac": "Motor spanwise station, fraction of semi-span",
        "servo_station": "Servo centre, fraction of local chord",
        "servo_span_frac": "Servo spanwise station, fraction of semi-span",
        "le_sweep_deg": "Leading-edge sweep, positive aft",
        "battery_station": "Battery forward face, fraction of root chord",
    }
    return [_variable(name, types[name], value, notes[name])
            for name, value in zip(am.DESIGN_VARS, values)]


def derived_variables(p):
    """Dimensions a sketch can consume without doing any arithmetic.

    Everything here is a length in millimetres from either the centreline
    (spanwise) or the root leading edge (chordwise), or an angle in degrees.
    The prefix keeps them visually separate from the raw design variables, which
    share a studio with them.

    Chordwise positions include the local leading-edge offset from sweep, so a
    sketch can place a feature directly rather than composing a sweep term with
    a chord term and getting the sign wrong.
    """
    g = am.unpack(p)
    r = am.evaluate(p)
    semi = 0.5 * am.SPAN

    root_chord = float(g["root_chord"])
    tip_chord = float(g["tip_chord"])
    mac = float(r["mac"])
    x_hinge = float(g["x_hinge"])

    def le_at(y):
        """Leading-edge offset aft of the root leading edge at spanwise y."""
        return float(am.leading_edge_x(abs(y) / semi, float(g["le_sweep_deg"])))

    def chord_at(y):
        return root_chord + (tip_chord - root_chord) * (abs(y) / semi)

    elevon_in = float(g["elevon_inboard_y"])
    elevon_out = float(g["elevon_outboard_y"])
    servo_y = float(g["servo_y"])
    servo_chord = float(g["servo_chord"])
    motor_y = float(g["motor_y"])

    # Elevon chord, which with a constant-chord elevon is one number for the
    # whole surface -- the reason the hinge line can be drawn parallel to the
    # trailing edge instead of being constructed station by station.
    elevon_chord = float(am.elevon_chord_at(mac, x_hinge, mac))

    entries = [
        ("cad_span", LENGTH, am.SPAN, "Full span, tip to tip"),
        ("cad_semi_span", LENGTH, semi, "Centreline to tip"),
        ("cad_root_chord", LENGTH, root_chord, "Root chord"),
        ("cad_tip_chord", LENGTH, tip_chord, "Tip chord"),
        ("cad_root_thickness", LENGTH, float(g["root_thickness"]),
         "Maximum root section thickness"),
        ("cad_tip_thickness", LENGTH, float(g["tip_thickness"]),
         "Maximum tip section thickness"),
        ("cad_mac", LENGTH, mac, "Mean aerodynamic chord"),
        ("cad_le_sweep", ANGLE, float(g["le_sweep_deg"]),
         "Leading-edge sweep, positive aft"),
        ("cad_te_sweep", ANGLE, float(r["te_sweep_deg"]),
         "Trailing-edge sweep, positive aft"),
        ("cad_c4_sweep", ANGLE, float(r["c4_sweep_deg"]),
         "Quarter-chord sweep, positive aft (reported, not chosen)"),
        ("cad_tip_le_offset", LENGTH, le_at(semi),
         "Tip leading edge aft of the root leading edge"),

        # Elevon.  Given as a chord and two spanwise stations rather than as a
        # hinge fraction, because that is how a constant-chord elevon is drawn:
        # one offset from the trailing edge, held across the whole surface.
        ("cad_elevon_chord", LENGTH, elevon_chord,
         "Elevon chord, constant across the span"),
        ("cad_elevon_inboard_y", LENGTH, elevon_in,
         "Elevon inboard end from the centreline"),
        ("cad_elevon_outboard_y", LENGTH, elevon_out,
         "Elevon outboard end from the centreline"),
        ("cad_elevon_span", LENGTH, elevon_out - elevon_in,
         "Elevon span, one side"),
        ("cad_elevon_tip_margin", LENGTH, semi - elevon_out,
         "Fixed wing outboard of the elevon, for the hinge anchor"),

        # Motors and props.
        ("cad_motor_y", LENGTH, motor_y, "Motor from the centreline"),
        ("cad_motor_x", LENGTH, le_at(motor_y),
         "Motor axis aft of the root leading edge, on the local leading edge"),
        ("cad_prop_diameter", LENGTH, am.PROP_DIAMETER, "Propeller diameter"),

        # Servo pocket.  Chordwise position is the box centre in its own
        # section, carried back to the root datum by the sweep offset.
        ("cad_servo_y", LENGTH, servo_y, "Servo centre from the centreline"),
        ("cad_servo_x", LENGTH,
         le_at(servo_y) + float(g["servo_station"]) * servo_chord,
         "Servo centre aft of the root leading edge"),
        ("cad_servo_length", LENGTH, am.SERVO_LENGTH, "Servo body, chordwise"),
        ("cad_servo_width", LENGTH, am.SERVO_WIDTH, "Servo body, spanwise"),
        ("cad_servo_depth", LENGTH, am.SERVO_DEPTH, "Servo body, through thickness"),
        ("cad_servo_chord", LENGTH, servo_chord, "Local chord at the servo station"),
        ("cad_pushrod_length", LENGTH, float(r["pushrod_length"]),
         "Servo output to the hinge line, chordwise"),

        # Battery pocket, on the centreline.
        ("cad_battery_x", LENGTH, float(g["battery_station"]) * root_chord,
         "Battery forward face aft of the root leading edge"),
        ("cad_battery_length", LENGTH, am.BATTERY_LENGTH, "Battery length"),
        ("cad_battery_thickness", LENGTH, am.BATTERY_THICKNESS,
         "Battery depth through the section"),

        # Shell.
        ("cad_wall_thickness", LENGTH, am.WALL_THICKNESS, "Printed wall thickness"),
    ]

    # Local chord at each elevon end, so the hinge line can be checked against
    # the trailing edge at both ends rather than assumed parallel.
    entries.append(("cad_chord_at_elevon_inboard", LENGTH, chord_at(elevon_in),
                    "Local chord at the elevon inboard end"))
    entries.append(("cad_chord_at_elevon_outboard", LENGTH, chord_at(elevon_out),
                    "Local chord at the elevon outboard end"))

    return [_variable(name, var_type, value, note)
            for name, var_type, value, note in entries]


def build_payload(p):
    """The full variable list this module writes to the studio."""
    return design_variables(p) + derived_variables(p)


# --- Reading and writing ------------------------------------------------------


def read_variables(element_id=None, document_id=DOCUMENT_ID,
                   workspace_id=WORKSPACE_ID):
    """Current contents of the variable studio."""
    element_id = element_id or studio_id()
    return _request(
        "GET",
        f"/variables/d/{document_id}/w/{workspace_id}/e/{element_id}/variables")


def verify(payload, element_id=None, document_id=DOCUMENT_ID,
           workspace_id=WORKSPACE_ID):
    """Read the studio back and confirm it holds what was sent.

    Worth a second request because the write path fails quietly: Onshape returns
    200 for a payload it only partly understands, and the first version of this
    module wrote 43 correctly named, correctly typed variables with empty
    expressions.  Nothing short of reading them back distinguishes that from
    success.

    Returns a list of human-readable differences, empty when the studio matches.
    """
    stored = {v["name"]: v
              for studio in read_variables(element_id, document_id, workspace_id)
              for v in studio.get("variables") or []}

    problems = []
    for sent in payload:
        got = stored.get(sent["name"])
        if got is None:
            problems.append(f"{sent['name']}: missing from the studio")
            continue
        if got.get("type") != sent["type"]:
            problems.append(
                f"{sent['name']}: type {sent['type']} -> {got.get('type')}")
        # Onshape normalizes whitespace and may echo the expression in a
        # canonical form, so compare loosely; an empty expression is the failure
        # this is really looking for.
        want = sent["expression"].replace(" ", "")
        have = (got.get("expression") or "").replace(" ", "")
        if not have:
            problems.append(f"{sent['name']}: expression is empty")
        elif have != want:
            problems.append(
                f"{sent['name']}: expression {sent['expression']!r}"
                f" -> {got.get('expression')!r}")

    for name in set(stored) - {v["name"] for v in payload}:
        problems.append(f"{name}: left over in the studio, not in the payload")

    return problems


def evaluate_in_part_studio(names, part_studio_id=AIRFRAME_STUDIO_ID,
                            document_id=DOCUMENT_ID, workspace_id=WORKSPACE_ID):
    """What a part studio actually resolves these variables to.

    The strongest available check, and a different one from :func:`verify`: that
    confirms the studio stores the expressions, this confirms a part studio
    inherits them and evaluates them to the numbers the model computed.  The
    variables endpoint reports ``value`` as null for every variable regardless,
    so it cannot answer this.

    Lengths come back in millimetres and angles in degrees, matching the units
    the expressions were written in.  A name the part studio cannot see raises
    from the API rather than returning a default, which is the desired failure:
    a silently missing variable is how a sketch ends up dimensioned off a stale
    number.
    """
    lookups = ",\n".join(
        f"        {name!r} : getVariable(context, {name!r})" for name in names)
    script = f"""
    function(context is Context, queries) {{
        var out = {{
{lookups}
        }};
        var scaled = {{}};
        for (var entry in out) {{
            var v = entry.value;
            if (v is ValueWithUnits) {{
                // Report in the units the expression was written in, so the
                // comparison against the model is direct.
                scaled[entry.key] = isLength(v) ? v / millimeter : v / degree;
            }} else {{
                scaled[entry.key] = v;
            }}
        }}
        return scaled;
    }}
    """
    response = _request(
        "POST",
        f"/partstudios/d/{document_id}/w/{workspace_id}/e/{part_studio_id}"
        "/featurescript",
        body={"script": script, "queries": {}})

    # The FeatureScript result is a tagged tree; flatten the one map level it
    # returns rather than carrying Onshape's serialization format outward.
    entries = response["result"]["value"]
    return {e["key"]["value"]: e["value"]["value"] for e in entries}


def push_variables(p=None, element_id=None, document_id=DOCUMENT_ID,
                   workspace_id=WORKSPACE_ID, dry_run=False):
    """Overwrite the variable studio from a design vector.

    ``p`` defaults to the cached optimum, falling back to the baseline, matching
    what the plots draw so the CAD and the figures cannot come from different
    design points.

    Returns the payload that was sent, so a caller can log or diff it.  With
    ``dry_run`` the payload is built and returned without a write, which is the
    cheap way to see what a model change did to the CAD dimensions.
    """
    if p is None:
        cached = opt.load_optimum()
        p = am.BASELINE if cached is None else cached

    payload = build_payload(p)
    if dry_run:
        return payload

    element_id = element_id or studio_id()
    _request(
        "POST",
        f"/variables/d/{document_id}/w/{workspace_id}/e/{element_id}/variables",
        body=payload)
    return payload


# --- CLI ----------------------------------------------------------------------


def _print_payload(payload):
    width = max(len(v["name"]) for v in payload)
    for v in payload:
        print(f"  {v['name']:<{width}}  {v['expression']:>14}"
              f"   {v['description']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="list the document's tabs and their element ids")
    parser.add_argument("--read", action="store_true",
                        help="print the variable studio's current contents")
    parser.add_argument("--check", action="store_true",
                        help="evaluate every variable in the Airframe part "
                             "studio and compare against the model")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be written without writing it")
    parser.add_argument("--baseline", action="store_true",
                        help="push the hand-picked baseline instead of the optimum")
    parser.add_argument("--element", default=None,
                        help="variable studio element id, overriding the lookup")
    args = parser.parse_args()

    if args.list:
        for element in elements():
            print(f"  {element.get('elementType', '?'):<16}"
                  f" {element.get('id')}   {element.get('name')}")
        return

    if args.read:
        print(json.dumps(read_variables(args.element), indent=2))
        return

    if args.check:
        payload = push_variables(am.BASELINE if args.baseline else None,
                                 dry_run=True)
        expected = {v["name"]: float(v["expression"].split()[0]) for v in payload}
        got = evaluate_in_part_studio(list(expected))

        width = max(len(n) for n in expected)
        worst = 0.0
        for name, want in expected.items():
            have = got[name]
            error = abs(have - want)
            worst = max(worst, error)
            flag = "" if error < 1e-3 else "   <-- MISMATCH"
            print(f"  {name:<{width}}  model {want:>12.4f}"
                  f"   CAD {have:>12.4f}{flag}")
        print(f"\nLargest difference {worst:.2e}"
              f" ({'match' if worst < 1e-3 else 'MISMATCH'})")
        raise SystemExit(0 if worst < 1e-3 else 1)

    p = am.BASELINE if args.baseline else None
    payload = push_variables(p, element_id=args.element, dry_run=args.dry_run)

    label = "baseline" if args.baseline else "optimum"
    if args.dry_run:
        print(f"Would write {len(payload)} variables ({label}):")
        _print_payload(payload)
        return

    print(f"Wrote {len(payload)} variables ({label}) to "
          f"{VARIABLE_STUDIO_NAME!r}:")
    _print_payload(payload)

    problems = verify(payload, element_id=args.element)
    if problems:
        print(f"\nVerification FAILED -- {len(problems)} problems:")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("\nVerified: the studio matches the payload.")


if __name__ == "__main__":
    main()
