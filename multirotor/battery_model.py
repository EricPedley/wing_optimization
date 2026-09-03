"""1S LiPo battery model: terminal voltage under load and hover flight time.

Scope is deliberately narrow -- three BetaFPV "LAVA II" 1S HV-LiPo packs
(480/580/680 mAh), the only batteries this build is choosing between. Each
pack's datasheet gives one bench discharge curve (voltage vs time) recorded
at a fixed current near the pack's rated max (see conversation: the three
attached charts are 480mAh@27A/56.3C, 580mAh@33A/56.9C, 680mAh@39A/57.4C --
all approximately the same C-rate, not a current sweep of one pack). There is
no per-pack multi-current data to fit sag against, so this uses a standard
Thevenin-equivalent model instead: a fixed open-circuit-voltage-vs-state-of-
charge curve (shape only, from SimITL's generic 1S LiPo curve --
~/code/SimITL/src/sim/physics.h -- since that shape already matches these
curves well; see fit below) minus a per-pack constant internal resistance,
V(soc, I) = OCV(soc) - I * r_int_ohm. r_int_ohm is fit per pack from its one
measured curve, converting elapsed bench-test time to state of charge via
mAh_drawn = current_a * t / 3600 -- this is standard for LiPo packs (roughly
current-independent over the range flown here) and is what lets voltage be
predicted at currents other than the one measured.

Fit: least-squares over digitized (time_s, voltage) points from the three
attached charts, holding OCV(soc) fixed to SimITL's curve and solving for the
one free parameter r_int_ohm per pack. Residual mse ~0.02-0.03 V^2 (~0.15-0.2V
RMS) against curves that themselves cross ~1.3V of range -- adequate for a
flight-time estimate, not for high-precision voltage prediction. Fitted
r_int_ohm comes out consistent across all three packs (9.8-10.8 mOhm) despite
being fit independently, which is the intended sanity check: these are the
same cell line, so similar internal resistance across capacities is
physically expected and its absence would have flagged a bad fit.
"""

import jax.numpy as jnp

# SimITL's generic 1S LiPo open-circuit-voltage curve: (fraction of rated
# capacity discharged, per-cell OCV). Negative x allows for the ~1-3% a pack
# is typically overcharged above its rated capacity; x > 1 lets voltage keep
# falling below the rated-capacity point instead of clamping, since this
# project deliberately runs packs past 100% "rated" discharge (see
# quad_model.hover_point's no-reserve-margin choice). Not our own fit -- the
# curve's *shape* (knee locations/slope) is carried over unmodified because it
# already tracked the three BetaFPV curves well once the constant IR term
# absorbed the rest (see module docstring); only r_int_ohm below is fit to
# BetaFPV-specific data.
_OCV_SOC_FRAC = jnp.array([-0.06, 0.0, 0.01, 0.04, 0.50, 0.60, 0.85, 1.0, 1.01, 1.03, 1.06, 1.08])
_OCV_VOLTS = jnp.array([4.4, 4.2, 4.05, 3.97, 3.85, 3.7, 3.63, 3.49, 3.4, 3.3, 3.0, 0.0])


def open_circuit_voltage(soc_frac_drawn):
    """No-load per-cell voltage at a given fraction of rated capacity drawn
    (0 = full, 1 = fully discharged to rated capacity; can exceed 1).

    jnp.interp clamps outside the table's range (below -0.06 or above 1.08)
    to the endpoint value rather than extrapolating -- fine here since 1.08
    (0V) is already past any useful discharge point.
    """
    return jnp.interp(soc_frac_drawn, _OCV_SOC_FRAC, _OCV_VOLTS)


# name -> (capacity_mah, mass_kg, r_int_ohm). r_int_ohm fit per-pack as
# described in the module docstring; capacity/mass are datasheet values from
# the conversation (BetaFPV LAVA II 1S HV-LiPo: 480/580/680mAh, 12.6/14.1/
# 16.2g).
BATTERIES = {
    "480mAh": dict(capacity_mah=480.0, mass_kg=12.6e-3, r_int_ohm=9.82e-3),
    "580mAh": dict(capacity_mah=580.0, mass_kg=14.1e-3, r_int_ohm=10.84e-3),
    "680mAh": dict(capacity_mah=680.0, mass_kg=16.2e-3, r_int_ohm=10.26e-3),
}


def terminal_voltage(soc_frac_drawn, current_a, r_int_ohm):
    """Per-cell (1S, so pack) voltage under load: OCV minus the Ohmic sag
    from a constant internal resistance. Can go non-physically negative at
    combined extremes (deep discharge + high current) since neither term is
    clamped -- callers doing a flight-time integration should stop at a
    voltage floor (e.g. 3.0V) rather than trusting this near/below zero.
    """
    return open_circuit_voltage(soc_frac_drawn) - current_a * r_int_ohm


def capacity_mah(battery_name):
    return BATTERIES[battery_name]["capacity_mah"]


def mass_kg(battery_name):
    return BATTERIES[battery_name]["mass_kg"]


def r_int_ohm(battery_name):
    return BATTERIES[battery_name]["r_int_ohm"]
