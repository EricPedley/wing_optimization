# Verification

- Run Python commands with `uv run`.
- In environments with ROS installed, disable unrelated pytest plugin autoload: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest multirotor/test_prop_factor_graph.py -q`. Otherwise ROS's Python 3.10 pytest plugins can fail during collection in this project's Python 3.13 environment.
- Run the bench inference report with `uv run python -m multirotor.prop_factor_graph`.
- Some multirotor tests require the external SimITL catalogue at `~/code/SimITL` and skip when it is absent.
