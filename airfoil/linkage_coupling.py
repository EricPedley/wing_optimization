"""Boundary between the airfoil model and the linkage model.

The two models were developed separately and work in different units: the
airfoil model is strictly SI metres, the linkage model millimetres.  This
module is the only place they meet, so the conversion lives here once rather
than being scattered through the geometry code where a missing factor of a
thousand would be easy to write and hard to see.

It also fixes what used to be a hardcoded constant.  Before this, the linkage
was optimized on its own to maximize mechanical advantage and the answer was
pasted into the airfoil model as ``LINKAGE_ADVANTAGE``.  That was wrong twice
over: the advantage was measured at a servo position the airfoil geometry does
not actually put the servo at, and maximizing advantage is the wrong objective
anyway.  The servo's stroke is a fixed budget spent on either torque or throw --
``advantage * angle_span`` is very nearly ``servo_travel``, which is just
virtual work -- so more advantage always costs deflection range.  Only the
coupled problem knows which side of that trade to be on.
"""

import jax.numpy as jnp

from linkage import linkage_model as lm

# The one place millimetres become metres.
MM_PER_M = 1000.0


def linkage_metrics(pushrod_length_m, servo_rod_y, servo_travel_mm,
                    flap_x_mm, flap_y_mm):
    """Linkage performance at one airfoil design point.

    ``pushrod_length_m`` is the chordwise hinge-to-servo run in metres, which is
    the linkage model's ``servo_x``.  It is derived from the airfoil geometry
    rather than chosen, because the servo's chordwise station and the hinge
    station are both already design variables and the distance between them is
    not free to disagree with them.

    ``servo_rod_y`` is the linkage model's ``servo_y``: where the control rod
    attaches on the servo end, offset perpendicular to the hinge axis.  It is
    *not* the airfoil model's ``servo_y``, which is a spanwise distance from the
    centreline, and it is not the servo body's own depth either -- the body sits
    at ``servo_height`` and the rod picks up a short way off it.

    All millimetre arguments stay millimetres; only the returned lengths are
    converted, so a caller never has to know what units the linkage model uses.
    """
    servo_x_mm = pushrod_length_m * MM_PER_M
    m = lm.metrics(jnp.array([servo_x_mm, servo_rod_y, servo_travel_mm,
                              flap_x_mm, flap_y_mm]))
    return {
        # Worst mechanical advantage over the stroke, in metres of hinge torque
        # per newton of servo force.  The worst point rather than the mean or
        # the peak, because the servo has to drive the surface everywhere in its
        # travel and it is the weakest point that decides whether it stalls.
        "advantage": m["min"] / MM_PER_M,
        # Half the sweep, because the airfoil model's deflection is a
        # single-sided amplitude while angle_span is peak to peak.
        "delta_max_deg": 0.5 * jnp.degrees(jnp.abs(m["angle_span"])),
        # Feasibility.  A linkage that cannot close reports a *larger*
        # advantage, so without these the optimizer is rewarded for picking
        # mechanisms that cannot be built.
        "violation": m["violation"],
        "valid_fraction": m["valid_fraction"],
        "deadness": m["deadness"],
        "rod_length": m["rod_length"] / MM_PER_M,
    }
