# Handoff: propeller/motor joint inference as a factor graph

Supersedes the diagnosis in `PROP_AERO_MODELING_HANDOFF.md` (that document's
*symptoms* are all real and still worth reading; its *root-cause attribution*
to Reynolds-number effects is wrong -- see §1).

## TL;DR

- The BEMT model's dominant error is **structural**: it forces
  `CT ∝ (P/D)^1.0` because it assumes an uncambered blade. Real data says
  `CT ∝ (P/D)^0.30`. Two independent estimators agree (§1).
- Reynolds effects are real but **~6% per rpm doubling** -- an order of
  magnitude too small to be the main term, and the cross-prop correlation has
  the **opposite sign** from the Re hypothesis (§1).
- Much of the "noise" in the old handoff was a **pooling artifact**. Fit per
  bench series and the data is clean (r² > 0.99 nearly everywhere) (§2).
- `tmotor_m1103_throttle_sweep.csv` is corrupt well beyond the one flagged
  row, and there is a cheap physics gate that finds the bad series (§2).
- `multirotor/prop_factor_graph.py` (committed, `a29bdca`) implements joint
  prop+motor inference in GTSAM. It removes an energetically-impossible
  artifact of the old back-out approach and auto-detects the bad datasheet
  (§4).
- A global geometry regression was tried and is a **dead end for the stated
  goal** -- it is not in the repo. §6 explains why, and §7 specs the
  latent-variable model that should replace it.

## 1. Root cause of the BEMT misfit

`prop_aero_model.bemt_thrust_torque`'s blade-element term makes thrust vanish
linearly as pitch → 0, i.e. `CT ∝ P/D`. Measured on the repo's own bench data:

| estimator | P/D exponent |
|---|---|
| per-prop CT regression across 16 clean props | **0.30** |
| factor-graph hierarchical fit (independent method, §6) | **0.26 ± 0.13** |
| what BEMT structurally forces | **1.0** |

That is ~5.7σ from the model's assumed value. **Physical cause:** propeller
blades are cambered, and a cambered section produces lift at zero geometric
angle of attack, so real static thrust does not vanish at low pitch. The
model's `CL = CL_ALPHA·α` has no zero-lift-angle term.

This single mismatch explains every symptom the old handoff catalogued:
low-P/D props badly underpredicted, high-P/D overpredicted, and the
"1.4 → >50" per-prop `CL_ALPHA` spread (a multiplier absorbing a wrong
exponent). Values above 2π are the tell -- thin-airfoil theory caps a real
lift slope there.

### Why it is not Reynolds number

- **Magnitude**: within-prop thrust exponent across 26 clean series is
  n = 2.087 mean / 2.068 median / 0.109 std, range 1.939-2.360. That is
  **+6% CT per rpm doubling**, consistent with Deters/Ananda/Selig
  (AIAA 2014-2151) reporting ~10%. Second-order against 50-100% residuals.
- **Sign**: `corr(CT/blade, log Re) = -0.71` across props. The Re hypothesis
  predicts a *positive* correlation. The apparent cross-prop "Re trend" is a
  diameter trend in disguise (bigger prop → bigger chord → higher Re).

### Two things that do NOT fix it (both tested)

- **Chord.** `CHORD_TO_DIAMETER_RATIO` and `CL_ALPHA` are *exactly*
  degenerate -- refits at 0.10/0.12/0.14 give **bit-identical** residuals with
  `CL_ALPHA` sliding 2.71 → 2.26 → 1.94. Only the product is identifiable from
  static thrust. So `PROP_AERO_MODELING_HANDOFF.md`'s "next step 1" (measure
  real chord) cannot help on its own. Solving for the chord each prop would
  need gives absurdities: GF35mm-3 requires a **54mm blade width on a 35mm
  prop**, GF1608-3 requires 19.8mm on 40mm.
- **Diameter-scaled chord.** Fitting `chord_ratio = c·(D/D_ref)^k` returned
  k ≈ -0.10 and barely moved residuals (RMS 50.1%). The missing physics is in
  the *lift model*, not the *geometry*.

### Partial fix, measured

Adding a zero-lift-angle offset (`p_eff = p + α₀·0.75·π·D`), fit at
α₀ ≈ 5.4° with `cl_alpha = 2π`: RMS 50% → 44%, and the high-P/D trend is
corrected. The 40mm props remain ~50% under, so a diameter/Re term is still
needed **on top of** this, not instead of it. Not committed.

## 2. Data quality (do this before any fit)

### The pooling artifact

The old handoff's "n ranges 1.9-2.4, some props below 1.0" came from pooling
multiple bench series per prop (different motor, kV, voltage) into one
regression. Split by `(prop, motor, kv, file)` and nearly every series is a
clean power law, r² > 0.99, n ∈ [1.94, 2.36]. **The data is better than it
looked**, which is bad news for the model: clean data with systematic
residuals means structural error, not noise.

### The physics gate

Within a fixed-voltage series, shaft power must scale ~rpm³, so current must
scale ~rpm^2.5-3. Series far below that have a corrupt rpm column:

```python
rpm = np.array([r["rpm"] for r in series])
I   = np.array([r["current_a"] for r in series])
A = np.vstack([np.log(rpm), np.ones(len(rpm))]).T
c, *_ = np.linalg.lstsq(A, np.log(I), rcond=None)
r2 = 1 - np.sum((np.log(I) - A@c)**2) / np.sum((np.log(I) - np.log(I).mean())**2)
ok = c[0] >= 2.2 and r2 >= 0.97          # keeps 183/300 rows
```

Fails on `tmotor_m1103`: GF1635-3, GF1636-4, GF2015-2 give current ∝
rpm^0.9-1.4. **More rows are bad than the single flagged row.**

### What is NOT corrupt

All 8 datasheets respect the momentum-theory bound on *electrical* power
(FM 0.37-0.58). The small M0803II props are real data. Two prop labels have
no alias entry and are silently dropped by `load_bench_rows`:
`GF1636-3` (m1103) and `GF1636-4(?)` (m1104).

### Black-box polynomial baseline

Per series, degree-2 in thrust and degree-3 in power (3-4 params against
10-11 points -- comfortable, and the right functional form):

| | min | median | r² > 0.99 |
|---|---|---|---|
| thrust deg-2 | 0.9020 | 0.9991 | 35/36 |
| power deg-3 | 0.9589 | 0.9994 | 35/36 |

**7 of 8 datasheets** have every series above 0.99 on both. The sole failure
is m1103, and the failing series are the same ones the physics gate rejects --
two unrelated methods agreeing the data is bad. Useful as a data-quality gate
and as a smooth calibration target.

## 3. Graph connectivity (bench coverage)

Props and motor-variants as nodes, bench series as edges: 18 props +
12 motor-variants, 52 series → **3 connected components** (4 if you require
≥4 points per series).

| # | props | motors | character |
|---|---|---|---|
| 1 | 9 | 6 | mixed 40-76mm; F1203, F1404, M1103×2, M1104, M1106 |
| 2 | 5 | 3 | **all ≤40mm**; M0803II ×3 windings only |
| 3 | 4 | 3 | **all 63-76mm**; F1204×2, F1303 |

The partition is stratified by size -- exactly the axis the modeling problem
lives on. Component 2 (the whole small-prop regime) is reachable only through
one motor model, so any M0803II calibration offset is indistinguishable from a
real aerodynamic property of all five small props.

**Consequence**: each component carries an independent gauge freedom. This is
confirmed numerically in §4.

**Cheapest fix**: 2 bridging bench series. Hurricane 3018-2 (comp 1) and
HQProp T3x1.8x3 3018 (comp 3) are nearly the same prop (76mm, ~1.8in, 3-blade)
in different components -- one sweep of either on the other's motor closes it.

## 4. What is implemented: `prop_factor_graph.py`

Committed as `a29bdca`. Run: `uv run python -m multirotor.prop_factor_graph`.
Requires `gtsam==4.3a2` (already added to `pyproject.toml`; note plain
`gtsam` on PyPI is Mac-only at 4.0.3 -- the 4.3 alphas carry Linux x86_64
wheels and need `--prerelease=allow`).

**State**: prop nodes `[log kT, log kP]` with `T = kT·rpm²`,
`P_mech = kP·rpm³`; motor nodes `[log Kt_scale, log I0_scale]` as
multiplicative corrections on datasheet values.

**Factors**: thrust (unary on prop); current (**binary**, prop×motor -- this
is what makes it joint rather than a back-out); motor priors with
per-motor strength (`LOOSE_MOTORS = {"M1103"}` gets wider priors). Huber
robust kernels on all data factors.

Size: 300 rows → 612 factors, 60 variables. Error 1677.7 → 1208.6.

### Why joint inference was necessary

Backing mechanical power out of bench current with fixed datasheet (kV, I0) --
what `calibrate_prop_aero.fit_induced_power_factor` does -- treats the motor
spec as exact and dumps its error into the prop. Measured consequences:

- Same prop on different motors disagrees with itself: **median 15.6% in kT**
  (worst: Hurricane 3018-2 78.6%, GF63MM-3 76.0%), 12.8% in kP.
- Implied figure of merit **exceeds 1.0** (energetically impossible) on many
  series, worst **4.38** -- even though the raw electrical data is fine.
- Cause: the `I - I0` subtraction amplifies error where these props operate
  (M0803II has I0 = 1.5-2.3 A).

### Results

| | naive back-out | joint |
|---|---|---|
| max FM | 4.38 | **1.04** |
| median FM | 1.03 | **0.64** |
| props with FM > 1 | many | **1/18** |

In-sample thrust RMS: **17.7%** all data, **8.0%** excluding m1103
(current 14.8%). BEMT on the same rows is 44-50%.

Residual RMS by datasheet -- the graph **found the bad file unprompted**
(8 of the top 12 outliers are M1103):

```
m1103 41.3% | m1106 20.3% | m1104 13.6% | f1203 8.1%
f1204  7.9% | m0803ii 6.5% | f1303  5.6% | f1404 2.4%
```

### Gauge check (confirms §3)

Linearize with all priors removed and take the Hessian spectrum:

```
smallest eigenvalues: [7.5e-13, 1.6e-12, 1.9e-12, 0.397, 0.571, 0.928]
near-null directions: 3        <- one per connected component
with priors: smallest eigenvalue 5.22, near-null 0
```

The unary motor priors are what make the problem well-posed, not decoration.

### Known weaknesses

- **Leave-one-motor-out**: pooled 16.2% thrust / **43.2% current**. Predicting
  an unseen motor from its datasheet alone is poor.
- **I0 corrections are suspect**: every motor wants `I0 × 0.15-0.55`. A
  uniform pull across 12 independent motors is a missing term, almost
  certainly **ESC losses** (measurement is at the battery, upstream of the
  ESC) plus I0 being spec'd at low voltage. `corr(Kt, I0)` runs 0.15-0.49.
  **Add an explicit ESC efficiency variable before trusting motor corrections
  physically.**
- **Posterior/prior variance ratios**: Kt is informed (0.26-0.72), I0 often is
  not (0.72-0.97 -- near 1.0 means the data said nothing and you are reading
  back the prior). Reported per motor in `main()`.
- Prop uncertainty is small: kT σ median 0.018, kP σ median 0.066.

## 5. Reproduction

Diagnostic scripts from the session lived in the scratchpad and are **not
preserved**. The reusable pieces are the physics gate (§2) and the module
itself. Everything in §1-§4 is reproducible from `load_bench_rows()` plus
those snippets.

## 6. Dead end: global geometry regression (NOT in the repo)

Tried, measured, then deliberately removed -- `prop_factor_graph.py` is back
at the flat model. Recorded so it is not re-attempted blindly.

**Form**: propeller similarity laws give the diameter dependence exactly --
`kT = C_T·ρ·D⁴/3600`, `kP = C_P·ρ·D⁵/216000` (monotonic and zero-intercept by
construction, no inequality constraints needed). Then a *global* log-linear
trend with per-prop slack (σ = 0.22) treated as noise:

```
log C_T = t₀ + t₁·log(P/D) + t₂·log(B)
log C_P = q₀ + q₁·log(P/D) + q₂·log(B)
```

**What worked**: fitted `t₁ = 0.260 ± 0.126`, `t₂ = 0.756 ± 0.335`
(C_T0 = 0.1158); `q₁ = 0.362`, `q₂ = 0.837` (C_P0 = 0.0623). Gauge freedoms
dropped **3 → 1** (the geometry layer bridges the disconnected components).
The `t₁` value is the independent confirmation cited in §1. Blade-count
exponent below 1.0 is consistent with blade interference, but too uncertain to
lean on.

**What failed**: leave-one-prop-out (geometry-only prediction) is **40.1% RMS**
pooled and wildly uneven -- GF3028-3 10.0% and M12199-3 11.6%, but HQProp
T3x1.8x3 **114.5%** and Hurricane 3016 56.9%. Shrinkage benefit is negligible
(k=2 rows: 11.8% → 11.2%; k=5: 9.3% → 10.0% i.e. *worse*), because a single
`kT` against a clean rpm² curve is already pinned by 2 points -- there is
little variance for a prior to reduce.

**Why it is the wrong structure regardless of fit quality** (the actual
reason it was dropped): it makes per-prop deviation *slack noise* rather than
*estimated coordinates*, and it leaves the curve **shape** fixed (one
parameter per curve, exponents hard-wired at 2 and 3). That cannot express
what the design space needs.

## 7. Next step: latent-variable reduced-order model (SPEC)

**Goal**: each propeller is described by *known physical* parameters (D, P, B)
plus a handful of *estimated, non-physical* latent coefficients. Together they
determine `thrust(rpm)` and `power(rpm)`. The latents are extra **design-space
dimensions**, not noise -- the optimizer can move in them to describe
propellers that are not in the catalogue.

Proposed form (log space throughout; `rpm_ref = 30000`, cf.
`prop_aero_model.REFERENCE_RPM`):

```
T(rpm) = ρ·D⁴·(rpm/60)² · exp( β·log B + γ·log(P/D) + zT + sT·log(rpm/rpm_ref) )
P(rpm) = ρ·D⁵·(rpm/60)³ · exp( δ·log B + ε·log(P/D) + zP + sP·log(rpm/rpm_ref) )
```

- **Per-prop latent** `z = (zT, sT, zP, sP)`: level *and* shape for each curve.
  `sT`, `sP` let the exponent deviate from 2 and 3, which is exactly the
  per-prop Reynolds slope -- measured range `n - 2 ∈ [-0.06, +0.36]`, so this
  is a real effect the flat model cannot represent.
- **Global structural** `θ = (β, γ, δ, ε)`: shared across props, part of the
  functional form. Expect `γ ≈ 0.26-0.30` (§1), `β ≈ 0.76`.
- **Motor nodes**: unchanged from §4 (plus the ESC term from §4's weaknesses).
- **Design space** = `(D, P, B)` physical × `z` latent.

Identifiability notes:
- `z` and the global intercepts are degenerate. Either drop the intercepts or
  impose zero-mean priors on `z` (random-effects formulation). Suggested
  σ ≈ 0.3 on levels, ≈ 0.15 on shape.
- `D⁴`/`D⁵` extrapolate soundly; the `C_T` trend does **not** (§6). Constrain
  the optimizer to the empirical hull/covariance of fitted `z`, and treat
  anything outside the benched geometry range as provisional.

Validation to run:
1. In-sample RMS vs the flat model (should improve -- shape params add real
   freedom where the data supports it).
2. **PCA of fitted `z` across props** → effective dimensionality of the design
   space. If 4 latents collapse to ~2 directions, say so and reduce.
3. Whether `sT` correlates with diameter/Re (it should, if it is capturing the
   §1 Reynolds effect) -- that would let the optimizer extrapolate `sT` rather
   than treating it as free.
4. Re-run the gauge check; confirm `z` priors leave no near-null directions.

## 8. Standing warnings

- Do not trust `realize_efficient_design` / `pareto_frontier` absolute numbers
  (TWR, hover power, g/W) without cross-checking against bench data. The
  original >6 g/W result that started this is physically impossible; real parts
  in this class top out near 2.5-3 g/W.
- Do not add free parameters to a static-thrust-only fit without checking
  identifiability first -- §1's exact chord/CL_ALPHA degeneracy is the standing
  example, and §4's gauge check is the cheap way to test.
- Keep the discrete/continuous split: measured lookups (or this graph) for
  realize/report paths where absolute accuracy matters; the closed-form
  differentiable model for the gradient-based optimizer.
