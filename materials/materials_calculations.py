"""Compare filaments for thin-wall micro RC airplane structures.

Core question: is a thin wall of a strong/stiff filament (e.g. 0.2mm ABS)
better than a thick wall of LW-PLA (e.g. 0.4mm)?

The answer hinges on which failure mode governs. For monocoque skins and
spar webs in a micro airplane the loads are bending and torsion carried by
thin shells, so the governing modes are usually *stability* (local buckling,
wrinkling) rather than material yield. Those modes scale very differently
with wall thickness than tensile strength does:

    tensile / yield capacity    ~ sigma * t        (linear in t)
    plate bending stiffness     ~ E * t^3 / 12     (cubic in t)
    local buckling stress       ~ E * (t/b)^2      (quadratic in t/b)

Mass per unit area of a wall is rho * t, so the useful figures of merit are
"capacity per unit areal mass" at fixed geometry.

Data source: materials/data/polymaker_detailed_tds.csv, scraped from
https://wiki.polymaker.com (per-product Technical Data Sheets).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data" / "polymaker_detailed_tds.csv"


@dataclass
class Material:
    name: str
    density: float  # g/cm^3
    E: float  # Young's modulus X-Y, MPa
    E_bend: float  # bending modulus X-Y, MPa
    tensile: float  # tensile strength X-Y, MPa
    bending: float  # bending strength X-Y, MPa
    charpy: float  # notched Charpy X-Y, kJ/m^2

    @property
    def rho(self) -> float:
        """Density in kg/m^3."""
        return self.density * 1000.0


def load_materials() -> list[Material]:
    out = []
    with open(DATA) as f:
        for row in csv.DictReader(f):
            out.append(
                Material(
                    name=row["product"],
                    density=float(row["density_g_cm3"]),
                    E=float(row["youngs_modulus_xy_mpa"]),
                    E_bend=float(row["bending_modulus_xy_mpa"]),
                    tensile=float(row["tensile_strength_xy_mpa"]),
                    bending=float(row["bending_strength_xy_mpa"]),
                    charpy=float(row["charpy_notched_xy_kj_m2"]),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Specific properties (per unit mass) -- material-only, geometry-free
# ---------------------------------------------------------------------------


def specific_strength(m: Material) -> float:
    """sigma / rho. Governs tension-limited members (kN*m/kg)."""
    return m.tensile / m.density


def specific_stiffness(m: Material) -> float:
    """E / rho. Governs stretching-dominated stiffness."""
    return m.E / m.density


def specific_bending_plate(m: Material) -> float:
    """E^(1/3) / rho.

    Classic Ashby index for a *plate* of fixed planform loaded in bending
    with thickness free to vary. This is the index that decides "thin wall of
    stiff stuff" vs "thick wall of light stuff": because D ~ E*t^3, you can
    trade thickness against modulus, and mass ~ rho*t.
    """
    return m.E_bend ** (1.0 / 3.0) / m.density


def specific_buckling_panel(m: Material) -> float:
    """E^(1/2) / rho.

    Ashby index for a panel/column where the free variable is thickness and
    the failure mode is elastic buckling at fixed in-plane dimensions.
    """
    return m.E**0.5 / m.density


# ---------------------------------------------------------------------------
# Direct head-to-head at fixed wall thickness
# ---------------------------------------------------------------------------


def areal_mass(m: Material, t_mm: float) -> float:
    """Mass per unit wall area, g/m^2, for wall thickness t (mm)."""
    return m.rho * (t_mm / 1000.0)


def plate_bending_stiffness(m: Material, t_mm: float, nu: float = 0.35) -> float:
    """Flexural rigidity D = E t^3 / (12 (1-nu^2)), in N*mm."""
    t = t_mm
    return m.E_bend * t**3 / (12.0 * (1.0 - nu**2))


def buckling_stress(m: Material, t_mm: float, b_mm: float, k: float = 4.0,
                    nu: float = 0.35) -> float:
    """Critical local buckling stress of a flat panel, MPa.

    sigma_cr = k * pi^2 * E / (12 (1-nu^2)) * (t/b)^2
    k=4.0 is the classic simply-supported-all-round, uniaxial compression case.
    b is the unsupported panel width (rib/stringer spacing).
    """
    import math

    return (
        k
        * math.pi**2
        * m.E_bend
        / (12.0 * (1.0 - nu**2))
        * (t_mm / b_mm) ** 2
    )


def tensile_capacity(m: Material, t_mm: float) -> float:
    """In-plane tensile load per unit width, N/mm."""
    return m.tensile * t_mm


def report_head_to_head(
    baseline: str = "LW-PLA",
    t_baseline: float = 0.4,
    challengers: tuple[str, ...] = ("PolyLite ABS", "Polymaker ASA", "PolyLite PLA"),
    t_challenger: float = 0.2,
    b_mm: float = 20.0,
) -> None:
    """Compare a thick baseline wall against thin walls of stiffer filaments.

    Everything is normalized to equal wall *area*, i.e. the same part geometry
    printed with different perimeter widths.
    """
    mats = {m.name: m for m in load_materials()}
    base = mats[baseline]

    print(f"\nHead-to-head: {baseline} @ {t_baseline}mm  vs  others @ {t_challenger}mm")
    print(f"(unsupported panel width b = {b_mm}mm for buckling)\n")

    header = (
        f"{'material':<16}{'t':>5}{'areal g/m2':>12}{'D N*mm':>10}"
        f"{'buckl MPa':>11}{'tens N/mm':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = [(base, t_baseline)] + [(mats[c], t_challenger) for c in challengers]
    for m, t in rows:
        print(
            f"{m.name:<16}{t:>5.2f}{areal_mass(m, t):>12.1f}"
            f"{plate_bending_stiffness(m, t):>10.2f}"
            f"{buckling_stress(m, t, b_mm):>11.4f}"
            f"{tensile_capacity(m, t):>11.2f}"
        )

    print("\nRelative to baseline (>1 means the thin wall wins):")
    b_mass = areal_mass(base, t_baseline)
    b_D = plate_bending_stiffness(base, t_baseline)
    b_buck = buckling_stress(base, t_baseline, b_mm)
    b_tens = tensile_capacity(base, t_baseline)
    for c in challengers:
        m = mats[c]
        t = t_challenger
        print(
            f"  {m.name:<16} mass x{areal_mass(m, t) / b_mass:.2f}  "
            f"D x{plate_bending_stiffness(m, t) / b_D:.2f}  "
            f"buckling x{buckling_stress(m, t, b_mm) / b_buck:.2f}  "
            f"tension x{tensile_capacity(m, t) / b_tens:.2f}"
        )


def equal_mass_thickness(m: Material, base: Material, t_base: float) -> float:
    """Thickness of `m` that gives the same areal mass as `base` at t_base."""
    return t_base * base.rho / m.rho


def report_equal_mass(baseline: str = "LW-PLA", t_baseline: float = 0.4,
                      b_mm: float = 20.0) -> None:
    """Fair fight: give every material the same weight budget, then compare."""
    mats = load_materials()
    base = {m.name: m for m in mats}[baseline]

    print(f"\nEqual-mass comparison (budget = {baseline} @ {t_baseline}mm "
          f"= {areal_mass(base, t_baseline):.1f} g/m2)\n")
    header = (
        f"{'material':<16}{'t_eq mm':>9}{'D N*mm':>10}{'buckl MPa':>11}"
        f"{'tens N/mm':>11}{'charpy':>9}"
    )
    print(header)
    print("-" * len(header))

    ranked = sorted(
        mats,
        key=lambda m: plate_bending_stiffness(m, equal_mass_thickness(m, base, t_baseline)),
        reverse=True,
    )
    for m in ranked:
        t_eq = equal_mass_thickness(m, base, t_baseline)
        print(
            f"{m.name:<16}{t_eq:>9.3f}{plate_bending_stiffness(m, t_eq):>10.2f}"
            f"{buckling_stress(m, t_eq, b_mm):>11.4f}"
            f"{tensile_capacity(m, t_eq):>11.2f}{m.charpy:>9.1f}"
        )


def report_indices() -> None:
    """Geometry-free Ashby indices, each ranked independently."""
    mats = load_materials()

    print("\nMaterial indices (higher = better, thickness free to vary)\n")
    header = (
        f"{'material':<16}{'sigma/rho':>11}{'E/rho':>9}"
        f"{'E^(1/3)/rho':>13}{'E^(1/2)/rho':>13}"
    )
    print(header)
    print("-" * len(header))
    for m in sorted(mats, key=specific_bending_plate, reverse=True):
        print(
            f"{m.name:<16}{specific_strength(m):>11.1f}{specific_stiffness(m):>9.0f}"
            f"{specific_bending_plate(m):>13.2f}{specific_buckling_panel(m):>13.2f}"
        )
    print("\n  sigma/rho    -> tension-limited parts (pull-tests, cable anchors)")
    print("  E/rho        -> stretch-dominated stiffness")
    print("  E^(1/3)/rho  -> BENDING of a plate w/ free thickness  <-- skins")
    print("  E^(1/2)/rho  -> buckling of a panel w/ free thickness <-- webs")


if __name__ == "__main__":
    report_indices()
    report_head_to_head()
    report_equal_mass()
