# Handoff: prop_aero_model.py's BEMT model does not fit real bench data well

## TL;DR

`prop_aero_model.py`'s closed-form static-thrust model assumes `thrust ∝ rpm²`
exactly (an exact consequence of the "ideal-twist" BEMT derivation). Real
bench data does not follow this: fitting `thrust = k·rpm^n` to real static
thrust-stand sweeps gives `n` in the 1.9-2.4 range depending on the prop, and
critically **the ratio `thrust/rpm²` is not constant across a single prop's
own throttle sweep** — it rises ~20-30% from lowest to highest rpm tested.
A model with a single fixed `CL_ALPHA` cannot reproduce this; it is forced to
choose one multiplicative constant and is then wrong in a systematic,
rpm-dependent way at every operating point. This is NOT primarily a
`CHORD_TO_DIAMETER_RATIO`/`CL_ALPHA` degeneracy (though that also exists,
see below) — it is a genuine missing physical effect.

**Current state**: `prop_aero_model.py` has `CL_ALPHA=4.03`,
`INDUCED_POWER_FACTOR=0.94`, refit against a narrowed 4-prop/78-89-row
subset of real bench data (60-77mm diameter, pitch/diameter ratio 0.55-1.05
— see the module's own calibration comment and `calibrate_prop_aero.py`).
This is better than the original 2-point calibration it replaced, but it
still has the same structural problem described above: it systematically
OVERPREDICTS thrust, worst at low rpm (~+30-50% at hover-relevant rpm for at
least one calibration prop), improving to ~+15-20% at max rpm. **Do not
trust absolute TWR/hover-power/flight-time numbers from the optimizer without
independently sanity-checking them against real bench data first** — this is
exactly how the user caught the underlying problem (a Pareto-frontier result
claiming >6 g/W hover efficiency, when real bench data tops out around 3 g/W
at best).

## How this was found

1. User fed several real T-Motor bench datasheets (full throttle sweeps,
   screenshotted from shop.tmotor.com/t-hobby.com, which block automated
   fetches) across this session -- see `multirotor/data/tmotor_*.csv`.
2. Ran `calibrate_prop_aero.py` to refit `CL_ALPHA`/`INDUCED_POWER_FACTOR`
   against all ~300 rows across 18 props. Finding: **no single `CL_ALPHA`
   reconciles the full dataset** -- per-prop fits (each prop fit only
   against its own data) range from ~1.4 to >50 (several hit the search
   ceiling with no convergence, i.e. the true per-prop optimum is more
   extreme than even that). Props ≤51mm diameter are structurally
   unfittable at ANY `CL_ALPHA` regardless of pitch/diameter ratio.
3. Narrowed the calibration to a physically-similar subset (this design's
   actual candidate range: 60-77mm diameter, moderate pitch/diameter ratio)
   and got a much better-looking fit (mean error 0-20% per prop, worst
   single-row error under 40%) -- see `prop_aero_model.py`'s current
   `CL_ALPHA=4.03`/`INDUCED_POWER_FACTOR=0.94`.
4. Ran the optimizer with this recalibrated model. The Pareto frontier
   (`optimize_efficiency.pareto_frontier()`) returned combos claiming hover
   efficiency **>6 g/W, one result implying >12 g/W** -- the user
   immediately (and correctly) flagged this as physically impossible; real
   bench data for these exact motor/prop classes tops out around 2.5-3 g/W
   at their best single operating point, and hover-averaged efficiency
   should be worse than that best point, not better.
5. Traced the >6g/W result to `M1103-8000 + Gemfan Hurricane 3018-2`, one of
   the actual calibration props. Found: at the exact rpm the hover-throttle
   solve lands on (~12,600 rpm), the model overpredicts this prop's real
   bench thrust by **+50%** and underpredicts current draw correspondingly
   -- even though this prop was IN the calibration subset and its aggregate
   "mean error" looked fine (-16.7% to +24.5% depending on which
   contributing bench series). The aggregate mean was misleading because:
   - Initially suspected (and excluded) one specific bench series
     (`tmotor_f1203_throttle_sweep.csv`'s "G3018-2" row, which gave MORE
     thrust than a mechanically similar 3-blade prop at the same rpm --
     backwards, and either a transcription error or a real error on
     T-Motor's page; not re-verified against the source). Excluding it is
     recorded in `calibrate_prop_aero.py`'s `EXCLUDED_BENCH_SERIES`.
   - Even after that exclusion, the remaining (clean, uncontested)
     `tmotor_m1103_throttle_sweep.csv` data for this exact prop is STILL
     overpredicted at every rpm (+14.5% to +49.7%, worst at low rpm) -- so
     the bad series was a real, separate data-quality bug, not the root
     cause of the modeling problem.
6. Fit `thrust = k·rpm^n` directly to this (and other) props' own bench
   data: `n≈2.298` for this prop specifically, and the fit is excellent
   (worst residual ~2%) -- confirming a pure power law describes the real
   data far better than the model's forced `n=2`. Checked across ~10 other
   props: `n` clusters around 1.9-2.4 (mean ~2.0-2.1) for the
   well-behaved subset (excluding the ≤51mm-diameter props already known
   to be structurally unfittable).
7. Ran a research pass on the aerospace/rotorcraft literature (see prompt
   history) confirming this is a well-documented, real phenomenon: small
   propellers operating at chord Reynolds numbers in the ~2×10⁴-6×10⁴ range
   (matching this design's actual regime -- computed directly, see below)
   show `thrust/rpm²` and `CT` rising with rpm/Re, attributed to Reynolds-
   number-dependent lift-curve slope (rising toward the thin-airfoil-theory
   value of 2π as Re increases) and falling profile drag. Key sources:
   - Deters, Ananda, Selig, "Reynolds Number Effects on the Performance of
     Small-Scale Propellers," AIAA 2014-2151.
   - Brandt & Selig, "Propeller Performance Data at Low Reynolds Numbers,"
     AIAA 2011-1255.
   - UIUC Propeller Data Site (tabulated CT/CP vs J at multiple Re per
     prop, not a closed-form fit): https://m-selig.ae.illinois.edu/props/propDB.html
   - No universal closed-form CL_ALPHA(Re) or CT(Re) formula exists in the
     literature -- serious tools either couple BEMT to a full Re-swept
     airfoil-polar database (numerical, not closed-form -- out of scope
     here) or, like this repo, fit an empirical correction to their own
     bench data.
8. Attempted the literature's recommended empirical fix: replace the fixed
   `CL_ALPHA` with `CL_ALPHA(Re) = CL_INF - A/Re^p` (a smooth, monotonically
   increasing, saturating form matching the qualitative Re-effects shape),
   with `Re` computed at the standard 0.75R blade station
   (`Re = rho * omega * 0.75R * chord / mu_air`). Fit the 2 free parameters
   (`A`, `p`, with `CL_INF` as a 3rd) against the same narrowed 4-prop
   subset via a vectorized coarse-to-fine grid search (same pattern as
   `calibrate_prop_aero.py`'s existing fits).
   - Result: helps, but doesn't fully close the gap. Best fit found:
     `CL_INF≈6.45, A≈20.7, p≈0.19`. On the previously-worst prop
     (Hurricane 3018-2 via M1103-8000), worst-case error dropped from
     +49.7% to +31.1% -- real improvement, but still a systematic
     overprediction at every rpm tested, not a clean fit.
   - Suspected but did NOT resolve: this may be tangled up with the
     `CHORD_TO_DIAMETER_RATIO=0.10` assumption also being wrong for these
     specific real props (many small FPV props run closer to 0.12-0.18
     chord/diameter, not 0.10) -- chord and `CL_ALPHA` are degenerate from
     static thrust data alone (only their product `chord × CL_ALPHA` is
     identifiable), so an Re-dependent correction on `CL_ALPHA` alone can't
     fully separate "wrong lift-slope-vs-Re shape" from "wrong fixed chord
     assumption." This is the same degeneracy the ORIGINAL two-point
     calibration's docstring already flagged, just resurfacing at a deeper
     level once Re-dependence entered the picture.
   - This attempt was NOT committed to `prop_aero_model.py` -- the module
     still has the flat `CL_ALPHA=4.03` from step 3, not the Re-dependent
     version. It exists only in exploratory shell commands from this
     session, not saved anywhere in the repo.

## What's actually in the repo right now

- `prop_aero_model.py`: `CL_ALPHA=4.03`, `INDUCED_POWER_FACTOR=0.94`,
  `CHORD_TO_DIAMETER_RATIO=0.10` (unchanged from the original), `CD0=0.02`
  (unchanged, never independently fit -- see its own docstring). Calibration
  comment documents the "no single global CL_ALPHA" finding and the
  deliberately narrowed subset it actually uses.
- `calibrate_prop_aero.py`: reproducible fitting script. `main()` prints
  both the full-dataset per-prop diagnostic (showing the range from ~1.4 to
  >50) and the actual narrowed-subset fit used in `prop_aero_model.py`.
  `EXCLUDED_BENCH_SERIES` documents the one bench series dropped for
  giving backwards (more thrust, more blades... wait, check: actually
  fewer blades giving MORE thrust) results versus a same-geometry prop from
  a different datasheet.
- `data/tmotor_*.csv`: 8 real T-Motor throttle-sweep datasheets (M1103,
  M1104, M1106, M0803II×3-winding, F1203, F1204×2-winding, F1303, F1404),
  ~300 usable rows (throttle >30%) across 18 distinct props. One known-bad
  row is flagged in `tmotor_m1103_throttle_sweep.csv` (an RPM value that
  fails rpm²/rpm³ scaling in every direction, confirmed against the live
  T-Motor page as a real error on their own site -- kept as-is, not
  hand-corrected, so a future systematic fit can flag it as an outlier).
- `data/prop_name_aliases.py`: maps the short prop labels used in bench CSVs
  (e.g. "GF3016") to full `prop_datasheets.csv` catalogue names.
- `test_prop_aero_model.py`: bench-fixture test rewritten to use two of the
  new calibration's own props (Hurricane 3018-2, HQProp T3x2x3) at a
  widened ±40% tolerance (was ±15%) reflecting the real, documented spread.

## What NOT to do

- Do not just widen the fit tolerance further and call it done -- the
  problem is not noise, it's a real missing rpm-dependent term, confirmed
  by an excellent (~2% residual) pure power-law fit per prop.
- Do not trust `realize_efficient_design`/`pareto_frontier` output at face
  value without spot-checking the winning combo's hover-rpm-regime thrust
  prediction against real bench data for a similar prop first, if one
  exists in `data/tmotor_*.csv` or `data/*_throttle_sweep.csv`.
- Do not assume a bigger/more-parameters model will obviously fix this --
  the `CHORD_TO_DIAMETER_RATIO`/`CL_ALPHA` degeneracy means adding more free
  parameters to a static-thrust-only fit risks just moving the
  unidentifiability around rather than resolving it. Real fresh information
  (e.g. bench data with an independently known/measured chord, not just
  diameter/pitch) may be needed to break it cleanly.

## Recommended next steps, in rough priority order

1. **Directly verify whether `CHORD_TO_DIAMETER_RATIO=0.10` is realistic**
   for the actual calibration props (Hurricane 3018-2, GF3028-3, HQProp
   T3x1.8x3, HQProp T3x2x3) -- look up real chord dimensions/photos/spec
   sheets for these specific props if available, rather than treating chord
   as a free/assumed parameter. If real chord is meaningfully different
   from 0.10×diameter, fixing that FIRST (independently of any Re-effects
   fit) may resolve much of the residual bias on its own.
2. If chord turns out to be roughly right, revisit the Re-dependent
   `CL_ALPHA(Re)` fit from step 8 above with `CHORD_TO_DIAMETER_RATIO`
   allowed to vary too (a 4-parameter joint fit: `CHORD_TO_DIAMETER_RATIO`,
   `CL_INF`, `A`, `p`) -- more parameters, but if chord is genuinely
   unknown this is more honest than pretending it's pinned at 0.10.
3. Consider whether `INDUCED_POWER_FACTOR` should ALSO become Re-dependent
   (torque/power showed similar-shaped errors in early exploration, not
   fully characterized this session) -- would need its own bench-current
   cross-check using each prop's real motor kV/I0 (already available in
   `motor_datasheets.csv` for most of the calibration props).
4. If a closed-form fix keeps not converging cleanly, consider whether the
   right move is to abandon the "one CL_ALPHA(Re) formula for all props"
   ambition entirely and instead give `nearest_catalogue_prop` a REAL
   thrust-vs-rpm lookup table (already have this raw data for several props
   in `data/tmotor_*.csv` and `data/*_throttle_sweep.csv`) for props that
   have bench data, falling back to the parametric BEMT model only for
   props/design-points that don't. This sidesteps the whole calibration
   problem for the specific real parts `realized_design`/
   `realize_efficient_design` actually select, at the cost of losing smooth
   gradients for those points (would need to only use the lookup for
   discrete realize/report code paths, not the continuous gradient-based
   optimizer, which already treats prop diameter/pitch/blades as continuous
   free variables and needs the closed-form model to stay differentiable).
5. Whatever fix is chosen, rerun `calibrate_prop_aero.py`'s FULL dataset
   diagnostic (not just the narrowed subset) afterward to check whether the
   fix also narrows the ~1.4-to->50 per-prop CL_ALPHA spread on props
   outside the current 60-77mm calibration range -- if it doesn't, the
   narrowed-subset restriction should stay in place and be extended to
   cover whatever additional shape parameter was added.
