# Verification

- Run Python commands with `uv run`.
- In environments with ROS installed, disable unrelated pytest plugin autoload: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest multirotor/test_prop_factor_graph.py -q`. Otherwise ROS's Python 3.10 pytest plugins can fail during collection in this project's Python 3.13 environment.
- Run the bench inference report with `uv run python -m multirotor.prop_factor_graph`.
- Some multirotor tests require the external SimITL catalogue at `~/code/SimITL` and skip when it is absent.
- Verify controls-model parity with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest multirotor/test_rate_plant.py -q` and inspect the nominal plant with `uv run python -m multirotor.rate_plant`. The controls model loads realized parts from the pinned `SimITL` submodule by default; `_unit(config_path=...)` accepts a quad config from another checkout.

# TODO

- Update the design optimizer's propulsion/battery assumptions separately: it still uses `DUTY_GAMMA=0.789`, `Kt=60/(2*pi*kV)`, and current-based `OCV-I*R` sag, while the controls model mirrors SimITL's linear PWM voltage mapping, `Kt=8.3/kV`, and mean-PWM/state-of-charge sag. Decide whether optimizer/simulator parity or the calibrated physical model is the target, then update and revalidate the optimizer, calibration, and design results together. Do not silently substitute the controls-only equations into the optimizer.
