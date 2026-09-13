import json

import numpy as np
import pytest

from multirotor.control_metrics import step_metrics, tracking_metrics, write_report


def _metrics(**overrides):
    inputs = dict(t=[0, 1, 2], reference=[0, 0, 0], rates=[0, 0, 0],
                  commands=[0, 0, 0], command_lower=-1, command_upper=1)
    inputs.update(overrides)
    return tracking_metrics(**inputs)


def test_zero_error_and_unknown_solver_metrics_are_json_serializable():
    result = _metrics()
    for key in ("rms_error_deg_s", "iae_deg", "peak_error_deg_s", "command_effort",
                "command_total_variation", "saturation_fraction", "saturation_duration_s"):
        assert result[key] == 0
    for key in ("max_solver_wall_ms", "p95_solver_wall_ms", "deadline_miss_fraction", "solver_failure_count"):
        assert result[key] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("sign", [-1, 1])
def test_error_units_sign_and_irregular_time_quadrature(sign):
    result = _metrics(t=[5, 6, 9], rates=np.deg2rad(sign * np.array([0, 2, 4])))
    assert result["rms_error_deg_s"] == pytest.approx(np.sqrt(8))
    assert result["iae_deg"] == pytest.approx(10)
    assert result["peak_error_deg_s"] == pytest.approx(4)


def test_single_axis_vector_and_column_shapes_agree():
    vector = _metrics(rates=np.deg2rad([1, 2, 3]), commands=[0, 0.5, 1])
    column = _metrics(reference=np.zeros((3, 1)), rates=np.deg2rad([[1], [2], [3]]),
                      commands=[[0], [0.5], [1]])
    assert vector == column


def test_multiple_axes_have_scalar_aggregates():
    result = _metrics(t=[0, 1, 4], reference=np.zeros((3, 2)),
                      rates=np.deg2rad([[3, -4], [3, -4], [3, -4]]),
                      commands=[[0, 0.5], [1, -0.5], [0.5, 0.5]],
                      command_lower=[-1, -0.5], command_upper=[1, 0.5])
    assert result["rms_error_deg_s"] == pytest.approx(np.sqrt(12.5))
    assert result["iae_deg"] == pytest.approx(28)
    assert result["peak_error_deg_s"] == pytest.approx(4)
    assert result["command_effort"] == pytest.approx(4)
    assert result["command_total_variation"] == pytest.approx(3.5)
    assert result["saturation_fraction"] == 1
    assert result["saturation_duration_s"] == 4
    json.dumps(result, allow_nan=False)


def test_commands_are_held_over_actual_intervals_and_saturation_is_any_axis():
    result = _metrics(t=[0, 1, 4], commands=[[0.5, 0], [1, 0], [0, 0]])
    assert result["command_effort"] == pytest.approx(3.25)
    assert result["command_total_variation"] == pytest.approx(1.5)
    assert result["saturation_duration_s"] == 3
    assert result["saturation_fraction"] == 0.75


def test_interval_commands_and_endpoint_commands_have_equal_integrals():
    a = _metrics(t=[0, 1, 4], commands=[-1, 0.5])
    b = _metrics(t=[0, 1, 4], commands=[-1, 0.5, 0.5])
    assert a == b
    assert a["command_effort"] == pytest.approx(1.75)
    assert a["saturation_fraction"] == 0.25


def test_saturation_includes_commands_outside_bounds_but_not_nearby_values():
    result = _metrics(t=[0, 1, 4], commands=[-1.01, 0.999, 1.01])
    assert result["saturation_duration_s"] == 1
    assert result["saturation_fraction"] == 0.25


def test_solver_statistics_explicit_deadline_and_failure_count():
    result = _metrics(solve_times_s=[0.001, 0.002, 0.003], solver_success=[True, False, True], dt=0.002)
    assert result["max_solver_wall_ms"] == 3
    assert result["p95_solver_wall_ms"] == pytest.approx(2.9)
    assert result["deadline_miss_fraction"] == pytest.approx(1 / 3)
    assert result["solver_failure_count"] == 1


@pytest.mark.parametrize("times,expected", [([0.011, 0.019], 0.5), ([0.011, 0.019, 0.021], 2 / 3)])
def test_solver_deadlines_inferred_from_irregular_intervals(times, expected):
    result = _metrics(t=[0, 0.01, 0.03], solve_times_s=times)
    assert result["deadline_miss_fraction"] == pytest.approx(expected)


def test_solver_success_can_be_reported_without_timings():
    result = _metrics(solver_success=[0, 1])
    assert result["solver_failure_count"] == 1
    assert result["deadline_miss_fraction"] is None


@pytest.mark.parametrize("overrides", [
    {"t": [0]}, {"t": [0, 0, 0]}, {"t": [0, 1, 1]}, {"t": [0, 2, 1]},
    {"t": [[0, 1, 2]]}, {"t": [0, 1, np.nan]}, {"t": [0, 1, np.inf]},
    {"reference": [0, 0]}, {"reference": [0, np.nan, 0]},
    {"rates": np.zeros((3, 2))}, {"rates": [0, 0, np.inf]}, {"rates": 0},
    {"commands": [0]}, {"commands": np.zeros((3, 0))},
    {"commands": np.zeros((3, 1, 1))}, {"commands": [0, np.nan, 0]},
    {"command_lower": 1}, {"command_lower": 2}, {"command_upper": np.inf},
    {"command_lower": [-1, -1]}, {"command_lower": [[-1]]},
    {"solve_times_s": [-0.1, 0]}, {"solve_times_s": [0, np.nan]},
    {"solve_times_s": [0]}, {"solve_times_s": [[0], [0]]},
    {"solver_success": [1, 2]}, {"solver_success": [1, np.nan]},
    {"solve_times_s": [0, 0], "solver_success": [1, 1, 1]},
    {"dt": 0}, {"dt": -1}, {"dt": np.nan}, {"dt": np.inf}, {"dt": [1]},
])
def test_invalid_tracking_inputs_are_rejected(overrides):
    with pytest.raises(ValueError):
        _metrics(**overrides)


@pytest.mark.parametrize("initial,target", [(0, 1), (0, -1), (2, 1), (-2, -1)])
def test_signed_step_rise_overshoot_and_settling(initial, target):
    t = np.arange(10, 18, dtype=float)
    reference = np.r_[initial, np.full(7, target)]
    progress = np.array([0, 0, 0.1, 0.9, 1.2, 1, 1, 1])
    rates = initial + (target - initial) * progress
    result = step_metrics(t, reference, rates)
    assert result["step_onset_s"] == 1
    assert result["rise_time_s"] == pytest.approx(1)
    assert result["overshoot_percent"] == pytest.approx(20)
    assert result["settling_time_s"] == pytest.approx(3.9)
    json.dumps(result, allow_nan=False)


def test_step_rise_interpolation_and_onset_are_not_tied_to_time_zero():
    result = step_metrics([10, 12, 13, 15, 16, 18], [0, 0, 2, 2, 2, 2], [0, 0, 0, 1, 2, 2])
    assert result["step_onset_s"] == 3
    assert result["rise_time_s"] == pytest.approx(2.4)
    assert result["settling_time_s"] == pytest.approx(2.96)
    assert result["overshoot_percent"] == 0


@pytest.mark.parametrize("rates", [[0, 0, 0, 0, 0], [0, 0, 0.2, 0.5, 0.7], [0, 0, -1, -2, -3]])
def test_never_attained_thresholds_are_none_not_argmax_zero(rates):
    result = step_metrics(np.arange(5), [0, 1, 1, 1, 1], rates)
    assert result["rise_time_s"] is None
    assert result["settling_time_s"] is None
    assert result["overshoot_percent"] == 0


def test_immediate_response_really_can_have_zero_rise_and_settling():
    result = step_metrics([0, 1, 2, 3], [0, 1, 1, 1], [0, 1, 1, 1])
    assert result["rise_time_s"] == 0
    assert result["settling_time_s"] == 0


@pytest.mark.parametrize("reference", [
    [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0], [0, 1, 1, 2, 2], [0, 0.25, 0.5, 0.75, 1],
])
def test_no_step_no_reference_plateau_or_multiple_steps_have_no_single_step_metrics(reference):
    result = step_metrics(np.arange(5), reference, reference)
    assert all(value is None for value in result.values())


def test_one_terminal_in_band_sample_is_not_a_settled_plateau():
    result = step_metrics(np.arange(5), [0, 1, 1, 1, 1], [0, 0, 0.1, 0.9, 1])
    assert result["rise_time_s"] == pytest.approx(1)
    assert result["settling_time_s"] is None


def test_settling_requires_remaining_inside_band_after_last_excursion():
    result = step_metrics(np.arange(8), [0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0.9, 1, 1, 1])
    assert result["settling_time_s"] == pytest.approx(3.8)


def test_step_metrics_multiple_axes_match_individual_calls():
    t = np.arange(6)
    reference = np.array([[0, 2], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    rates = np.array([[0, 2], [0, 2], [0.1, 1.9], [0.9, 1.1], [1, 1], [1, 1]])
    result = step_metrics(t, reference, rates)
    for axis in range(2):
        expected = step_metrics(t, reference[:, axis], rates[:, axis])
        for key, value in expected.items():
            assert result[key][axis] == value
    assert step_metrics(t, reference[:, :1], rates[:, :1]) == step_metrics(t, reference[:, 0], rates[:, 0])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("t,reference,rates", [
    ([0, 0], [0, 1], [0, 1]), ([0], [0], [0]),
    ([0, 1], [0, 1], [0, np.nan]), ([0, 1], [0, 1], [[0, 0], [1, 1]]),
])
def test_invalid_step_inputs_are_rejected(t, reference, rates):
    with pytest.raises(ValueError):
        step_metrics(t, reference, rates)


def test_overflowed_tracking_and_step_metrics_are_rejected():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            _metrics(rates=[1e308, 1e308, 1e308])
        with pytest.raises(ValueError):
            step_metrics([0, 1, 2, 3], [-1e308, 1e308, 1e308, 1e308], [0, 0, 0, 0])
        with pytest.raises(ValueError):
            step_metrics([0, 1, 2, 3], [0, 1, 1, 1], [0, 1e308, 1e308, 1e308])


def _report_data(timing=True):
    row = {"label": "MPC <script>alert('unsafe')</script>", "horizon": np.int64(20),
           "rms_error_deg_s": np.float64(2.5), "command_effort": 0.5, "solver_failure_count": None}
    trace = dict(label=row["label"], t=np.array([0, 0.01, 0.03]),
                 reference=np.zeros((3, 2)), rates=np.ones((3, 2)),
                 commands=np.array([[0.5, 0.6], [0.7, 0.8]]))
    if timing:
        trace["solve_times_s"] = np.array([0.001, 0.002])
    return [row], [trace]


@pytest.mark.parametrize("timing", [False, True])
def test_report_is_self_contained_escaped_and_contains_configuration(tmp_path, timing):
    rows, traces = _report_data(timing)
    output = tmp_path / "report.html"
    assert write_report(rows, traces, output, title="Rate <control> & report") == output
    document = output.read_text(encoding="utf-8")
    assert "plotly.js v" in document
    assert "Plotly.newPlot" in document
    assert '<script src="' not in document
    assert "Rate &lt;control&gt; &amp; report" in document
    assert "<script>alert('unsafe')</script>" not in document
    assert "&lt;script&gt;" in document
    assert "horizon" in document and "20" in document
    assert "Tracking versus command effort" in document
    assert "Angular rate tracking" in document
    assert ("Solver wall time" in document) == timing
    assert "Metrics and configuration" in document
    assert rows[0]["label"].startswith("MPC <script>")
    assert traces[0]["commands"].shape == (2, 2)


def test_report_refuses_to_overwrite_existing_file(tmp_path):
    rows, traces = _report_data()
    output = tmp_path / "report.html"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_report(rows, traces, output)
    assert output.read_text(encoding="utf-8") == "keep me"


def test_report_accepts_single_axis_endpoint_commands(tmp_path):
    rows, traces = _report_data(False)
    traces[0].update(reference=np.zeros(3), rates=np.ones(3), commands=np.array([0, 0.5, 1]))
    output = tmp_path / "single-axis.html"
    write_report(rows, traces, output)
    assert output.is_file()


def test_report_empty_data_is_supported(tmp_path):
    output = tmp_path / "empty.html"
    write_report([], [], output)
    assert output.is_file()


@pytest.mark.parametrize("change", [
    {"rms_error_deg_s": np.nan}, {"command_effort": None}, {"command_effort": -1},
    {"rms_error_deg_s": "bad"}, {"config": {"horizon": 5}},
])
def test_invalid_report_rows_do_not_create_output(tmp_path, change):
    rows, traces = _report_data()
    rows[0].update(change)
    output = tmp_path / "report.html"
    with pytest.raises(ValueError):
        write_report(rows, traces, output)
    assert not output.exists()


def test_invalid_report_trace_does_not_create_output(tmp_path):
    rows, traces = _report_data()
    traces[0]["solve_times_s"] = np.array([-1, 0])
    output = tmp_path / "report.html"
    with pytest.raises(ValueError):
        write_report(rows, traces, output)
    assert not output.exists()
