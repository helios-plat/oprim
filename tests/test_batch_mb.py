"""Tests for Mneme M-B batch: 11 math-solving elements.

Mandatory tests:
- CONIC_CASES: 10 sample conic problems
- test_verify_no_llm_call: verify_step has NO LLM dependency

Version: oprim v3.4.0
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# ── M-B imports ──────────────────────────────────────────────────────────────
from oprim.solve_conic import solve_conic, ConicParams, _classify_by_discriminant
from oprim.solve_function import solve_function, FunctionSolveInput
from oprim.solve_derivative import solve_derivative, DerivativeSolveInput
from oprim.solve_geometry3d import solve_geometry3d, Geometry3DInput
from oprim.solve_sequence import solve_sequence, SequenceSolveInput
from oprim.solve_trig import solve_trig, TrigSolveInput
from oprim.solve_probability import solve_probability, ProbabilitySolveInput
from oprim.verify_step import verify_step, StepVerifyInput, verify_steps
from oprim.kernel_to_plot2d import kernel_to_plot2d, Plot2DRequest, solve_result_to_plot2d
from oprim.kernel_to_three import kernel_to_three, Plot3DRequest
from oprim.generate_svg_diagram import generate_svg_diagram, generate_svg_from_plot2d, SVGConfig
from oprim.types import SolveResult, SolveStep, Plot2DData, Three3DData, SocraticTurnResult


# ─────────────────────────────────────────────────────────────────────────────
# MANDATORY: test_verify_no_llm_call
# CI checks that verify_step.py has NO import of ProviderRegistry/anthropic.
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyStepNoLLM:
    def test_verify_no_llm_call(self):
        """verify_step.py must not import ProviderRegistry, anthropic, or provider_registry."""
        verify_step_path = Path(__file__).parent.parent / "oprim" / "verify_step.py"
        source = verify_step_path.read_text()
        forbidden = ["ProviderRegistry", "anthropic", "obase.provider_registry"]
        for token in forbidden:
            assert token not in source, (
                f"verify_step.py must NOT import '{token}' — CI red line violation"
            )

    def test_verify_step_is_deterministic(self):
        """verify_step should never call any external service."""
        inp = StepVerifyInput(
            step_number=1,
            before_lhs="x**2 - 4",
            before_rhs="0",
            after_lhs="(x - 2)*(x + 2)",
            after_rhs="0",
            variable="x",
            description="Factor x²-4",
        )
        # If this calls an LLM it would fail (no API key in test env).
        result = verify_step(inp)
        assert result.is_correct is True

    def test_verify_step_catches_error(self):
        """Incorrect algebraic step should return is_correct=False."""
        inp = StepVerifyInput(
            step_number=1,
            before_lhs="x**2 - 4",
            before_rhs="0",
            after_lhs="(x - 2)*(x + 3)",  # wrong: should be (x+2)
            after_rhs="0",
            variable="x",
        )
        result = verify_step(inp)
        assert result.is_correct is False
        assert result.error_type == "algebraic_error"


# ─────────────────────────────────────────────────────────────────────────────
# CONIC_CASES — 10 sample conic problems
# ─────────────────────────────────────────────────────────────────────────────

CONIC_CASES = [
    # (expression, expected_type_substring)
    ("x**2 + y**2 - 9",                   "circle"),
    ("x**2 + y**2 - 2*x - 4*y - 4",       "circle"),
    ("x**2/4 + y**2/9 - 1",               "ellipse"),
    ("x**2/9 + y**2/4 - 1",               "ellipse"),
    ("x**2 - y**2/4 - 1",                 "hyperbola"),
    ("4*x**2 - y**2 - 16",                "hyperbola"),
    ("y**2 - 4*x",                        "parabola"),
    ("x**2 - 4*y",                        "parabola"),
    ("(x-1)**2 + (y+2)**2 - 25",          "circle"),
    ("x**2/16 + y**2/25 - 1",             "ellipse"),
]


class TestConicCases:
    @pytest.mark.parametrize("expression,expected_type", CONIC_CASES)
    def test_conic_classification(self, expression: str, expected_type: str):
        result = solve_conic(expression)
        assert result.solvable, f"Failed: {expression} — {result.error}"
        assert expected_type in result.answer, (
            f"Expression '{expression}' should be {expected_type}, got: {result.answer}"
        )

    def test_conic_returns_steps(self):
        result = solve_conic("x**2 + y**2 - 9")
        assert len(result.steps) >= 3

    def test_conic_circle_center_radius(self):
        result = solve_conic("x**2 + y**2 - 9")
        assert result.solvable
        # r = 3
        assert "3" in result.answer

    def test_conic_ellipse_radii(self):
        result = solve_conic("x**2/4 + y**2/9 - 1")
        assert result.solvable
        assert "ellipse" in result.answer

    def test_conic_invalid_expression(self):
        result = solve_conic("not_a_conic *** bad")
        assert not result.solvable
        assert result.error

    def test_conic_equality_format(self):
        result = solve_conic("x**2 + y**2 = 9")
        assert result.solvable
        assert "circle" in result.answer

    def test_discriminant_circle(self):
        ct = _classify_by_discriminant(1.0, 0.0, 1.0)
        assert ct == "circle"

    def test_discriminant_parabola(self):
        ct = _classify_by_discriminant(1.0, 0.0, 0.0)
        assert ct == "parabola"

    def test_discriminant_hyperbola(self):
        ct = _classify_by_discriminant(1.0, 0.0, -1.0)
        assert ct == "hyperbola"

    def test_discriminant_ellipse(self):
        ct = _classify_by_discriminant(2.0, 0.0, 3.0)
        assert ct == "ellipse"


# ─────────────────────────────────────────────────────────────────────────────
# solve_function
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveFunction:
    def test_zeros_quadratic(self):
        inp = FunctionSolveInput(expression="x**2 - 4", task="zeros")
        r = solve_function(inp)
        assert r.solvable
        assert "2" in r.answer or "-2" in r.answer

    def test_evaluate_at_point(self):
        inp = FunctionSolveInput(expression="x**2 + 1", task="evaluate", point=3.0)
        r = solve_function(inp)
        assert r.solvable
        assert "10" in r.answer

    def test_parity_even(self):
        inp = FunctionSolveInput(expression="x**2", task="parity")
        r = solve_function(inp)
        assert r.solvable
        assert "even" in r.answer

    def test_parity_odd(self):
        inp = FunctionSolveInput(expression="x**3", task="parity")
        r = solve_function(inp)
        assert r.solvable
        assert "odd" in r.answer

    def test_simplify(self):
        inp = FunctionSolveInput(expression="(x**2 - 1)/(x - 1)", task="simplify")
        r = solve_function(inp)
        assert r.solvable

    def test_compose(self):
        inp = FunctionSolveInput(
            expression="x**2 + 1",
            task="compose",
            g_expression="2*x",
        )
        r = solve_function(inp)
        assert r.solvable
        assert "4*x**2 + 1" in r.answer or "4" in r.answer

    def test_auto_selects_zeros_without_point(self):
        inp = FunctionSolveInput(expression="x - 5")
        r = solve_function(inp)
        assert r.solvable
        assert "5" in r.answer

    def test_monotonicity(self):
        inp = FunctionSolveInput(expression="x**3 - 3*x", task="monotonicity")
        r = solve_function(inp)
        assert r.solvable

    def test_inverse(self):
        inp = FunctionSolveInput(expression="2*x + 3", task="inverse")
        r = solve_function(inp)
        assert r.solvable


# ─────────────────────────────────────────────────────────────────────────────
# solve_derivative
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveDerivative:
    def test_first_derivative(self):
        inp = DerivativeSolveInput(expression="x**3 + 2*x", task="derivative")
        r = solve_derivative(inp)
        assert r.solvable
        assert "3*x**2" in r.answer or "3" in r.answer

    def test_second_derivative(self):
        inp = DerivativeSolveInput(expression="x**4", task="derivative", order=2)
        r = solve_derivative(inp)
        assert r.solvable
        assert "12*x**2" in r.answer or "12" in r.answer

    def test_critical_points(self):
        inp = DerivativeSolveInput(expression="x**3 - 3*x", task="critical_points")
        r = solve_derivative(inp)
        assert r.solvable
        assert "1" in r.answer or "-1" in r.answer

    def test_extrema_classification(self):
        inp = DerivativeSolveInput(expression="x**2 - 4*x + 3", task="extrema")
        r = solve_derivative(inp)
        assert r.solvable
        assert "local_min" in r.answer

    def test_inflection_points(self):
        inp = DerivativeSolveInput(expression="x**3 - 3*x", task="inflection")
        r = solve_derivative(inp)
        assert r.solvable
        assert "0" in r.answer

    def test_tangent_line(self):
        inp = DerivativeSolveInput(expression="x**2", task="tangent_line", point=2.0)
        r = solve_derivative(inp)
        assert r.solvable
        assert "4" in r.answer  # slope = 2x = 4 at x=2

    def test_auto_selects_derivative(self):
        inp = DerivativeSolveInput(expression="sin(x)")
        r = solve_derivative(inp)
        assert r.solvable
        assert "cos" in r.answer

    def test_trig_derivative(self):
        inp = DerivativeSolveInput(expression="sin(x)*cos(x)", task="derivative")
        r = solve_derivative(inp)
        assert r.solvable


# ─────────────────────────────────────────────────────────────────────────────
# solve_geometry3d
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveGeometry3D:
    def test_distance(self):
        inp = Geometry3DInput(task="distance", p1=(0, 0, 0), p2=(3, 4, 0))
        r = solve_geometry3d(inp)
        assert r.solvable
        assert "5" in r.answer

    def test_midpoint(self):
        inp = Geometry3DInput(task="midpoint", p1=(0, 0, 0), p2=(4, 6, 8))
        r = solve_geometry3d(inp)
        assert r.solvable
        assert "2" in r.answer and "3" in r.answer

    def test_sphere_volume(self):
        inp = Geometry3DInput(task="sphere", radius=3.0)
        r = solve_geometry3d(inp)
        assert r.solvable
        vol = (4/3) * math.pi * 27
        assert str(round(vol, 0))[:2] in r.answer  # check rough value

    def test_cylinder_volume(self):
        inp = Geometry3DInput(task="cylinder", radius=2.0, height=5.0)
        r = solve_geometry3d(inp)
        assert r.solvable
        vol = math.pi * 4 * 5
        assert "V" in r.answer

    def test_cone_volume(self):
        inp = Geometry3DInput(task="cone", radius=3.0, height=4.0)
        r = solve_geometry3d(inp)
        assert r.solvable
        assert "V" in r.answer

    def test_angle_planes(self):
        inp = Geometry3DInput(
            task="angle_planes",
            normal1=(1.0, 0.0, 0.0),
            normal2=(0.0, 1.0, 0.0),
        )
        r = solve_geometry3d(inp)
        assert r.solvable
        assert "90" in r.answer

    def test_distance_3d(self):
        inp = Geometry3DInput(task="distance", p1=(0, 0, 0), p2=(1, 1, 1))
        r = solve_geometry3d(inp)
        assert r.solvable
        dist = math.sqrt(3)
        assert f"{dist:.2f}"[:3] in r.answer or "1.73" in r.answer

    def test_missing_params_returns_error(self):
        inp = Geometry3DInput(task="sphere")  # no radius
        r = solve_geometry3d(inp)
        assert not r.solvable


# ─────────────────────────────────────────────────────────────────────────────
# solve_sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveSequence:
    def test_arithmetic_nth_term(self):
        inp = SequenceSolveInput(terms=[2, 5, 8, 11], task="nth_term", n=10)
        r = solve_sequence(inp)
        assert r.solvable
        assert "29" in r.answer  # a(10) = 2 + 9*3 = 29

    def test_geometric_nth_term(self):
        inp = SequenceSolveInput(terms=[2, 6, 18, 54], task="nth_term", n=5)
        r = solve_sequence(inp)
        assert r.solvable
        assert "162" in r.answer  # a(5) = 2*3^4 = 162

    def test_arithmetic_sum(self):
        inp = SequenceSolveInput(terms=[1, 2, 3, 4, 5], task="sum", count=5)
        r = solve_sequence(inp)
        assert r.solvable
        assert "15" in r.answer

    def test_geometric_sum(self):
        inp = SequenceSolveInput(terms=[1, 2, 4, 8], task="sum", count=4)
        r = solve_sequence(inp)
        assert r.solvable
        assert "15" in r.answer

    def test_type_check_arithmetic(self):
        inp = SequenceSolveInput(terms=[10, 7, 4, 1], task="type_check")
        r = solve_sequence(inp)
        assert r.solvable
        assert "arithmetic" in r.answer

    def test_type_check_geometric(self):
        inp = SequenceSolveInput(terms=[3, 9, 27], task="type_check")
        r = solve_sequence(inp)
        assert r.solvable
        assert "geometric" in r.answer

    def test_empty_terms_raises(self):
        inp = SequenceSolveInput(terms=[], task="type_check")
        r = solve_sequence(inp)
        assert not r.solvable

    def test_single_term_error(self):
        inp = SequenceSolveInput(terms=[5], task="nth_term", n=3)
        r = solve_sequence(inp)
        assert not r.solvable


# ─────────────────────────────────────────────────────────────────────────────
# solve_trig
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveTrig:
    def test_solve_sin_zero(self):
        inp = TrigSolveInput(expression="sin(x)", task="solve", rhs="0")
        r = solve_trig(inp)
        assert r.solvable
        assert "0" in r.answer

    def test_simplify_identity(self):
        inp = TrigSolveInput(expression="sin(x)**2 + cos(x)**2 - 1", task="simplify")
        r = solve_trig(inp)
        assert r.solvable
        assert "0" in r.answer

    def test_evaluate_sin_30(self):
        inp = TrigSolveInput(expression="sin(x)", task="evaluate", angle_degrees=30.0)
        r = solve_trig(inp)
        assert r.solvable
        # sin(30°) = 0.5
        assert "0.5" in r.answer or "1/2" in r.answer

    def test_evaluate_cos_90(self):
        inp = TrigSolveInput(expression="cos(x)", task="evaluate", angle_degrees=90.0)
        r = solve_trig(inp)
        assert r.solvable
        # cos(90°) ≈ 0
        val_str = r.answer
        assert "0" in val_str

    def test_period_sin(self):
        inp = TrigSolveInput(expression="sin(x)", task="period")
        r = solve_trig(inp)
        assert r.solvable
        assert "2*pi" in r.answer or "2π" in r.answer or "6.28" in r.answer

    def test_solve_cos_equation(self):
        inp = TrigSolveInput(expression="cos(x)", task="solve", rhs="1")
        r = solve_trig(inp)
        assert r.solvable

    def test_auto_selects_solve(self):
        inp = TrigSolveInput(expression="tan(x)", rhs="0")
        r = solve_trig(inp)
        assert r.solvable


# ─────────────────────────────────────────────────────────────────────────────
# solve_probability
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveProbability:
    def test_combinations(self):
        inp = ProbabilitySolveInput(task="combinations", n=5, k=2)
        r = solve_probability(inp)
        assert r.solvable
        assert "10" in r.answer

    def test_permutations(self):
        inp = ProbabilitySolveInput(task="permutations", n=5, k=2)
        r = solve_probability(inp)
        assert r.solvable
        assert "20" in r.answer

    def test_basic_probability(self):
        inp = ProbabilitySolveInput(task="basic", p_a=0.3, p_b=0.4)
        r = solve_probability(inp)
        assert r.solvable
        assert "0.12" in r.answer or "0.58" in r.answer

    def test_bayes(self):
        inp = ProbabilitySolveInput(
            task="bayes",
            p_a=0.01,          # P(disease)
            p_b_given_a=0.95,  # P(+|disease)
            p_b=0.05,          # P(+)
        )
        r = solve_probability(inp)
        assert r.solvable
        assert "0.19" in r.answer or "0.2" in r.answer

    def test_binomial(self):
        inp = ProbabilitySolveInput(task="binomial", n=10, k=3, p_success=0.5)
        r = solve_probability(inp)
        assert r.solvable
        # C(10,3) * 0.5^10 ≈ 0.1172
        assert "0.117" in r.answer or "0.118" in r.answer or "0.11" in r.answer

    def test_expected_value(self):
        inp = ProbabilitySolveInput(
            task="expected_value",
            values=[0.0, 1.0, 2.0],
            probabilities=[0.25, 0.5, 0.25],
        )
        r = solve_probability(inp)
        assert r.solvable
        assert "1" in r.answer  # E[X] = 0*0.25 + 1*0.5 + 2*0.25 = 1

    def test_combinations_auto(self):
        inp = ProbabilitySolveInput(n=6, k=3)
        r = solve_probability(inp)
        assert r.solvable
        assert "20" in r.answer

    def test_invalid_probabilities(self):
        inp = ProbabilitySolveInput(
            task="expected_value",
            values=[0.0, 1.0],
            probabilities=[0.3, 0.3],  # don't sum to 1
        )
        r = solve_probability(inp)
        assert not r.solvable


# ─────────────────────────────────────────────────────────────────────────────
# verify_step (additional)
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyStep:
    def test_valid_factoring(self):
        inp = StepVerifyInput(
            step_number=1,
            before_lhs="x**2 - 4",
            after_lhs="(x - 2)*(x + 2)",
            variable="x",
        )
        r = verify_step(inp)
        assert r.is_correct

    def test_invalid_factoring(self):
        inp = StepVerifyInput(
            step_number=1,
            before_lhs="x**2 - 4",
            after_lhs="(x - 2)*(x - 2)",  # wrong
            variable="x",
        )
        r = verify_step(inp)
        assert not r.is_correct

    def test_verify_expansion(self):
        inp = StepVerifyInput(
            step_number=2,
            before_lhs="(x + 1)**2",
            after_lhs="x**2 + 2*x + 1",
            variable="x",
        )
        r = verify_step(inp)
        assert r.is_correct

    def test_verify_steps_list(self):
        inputs = [
            StepVerifyInput(
                step_number=1,
                before_lhs="x**2 - 9",
                after_lhs="(x-3)*(x+3)",
                variable="x",
            ),
            StepVerifyInput(
                step_number=2,
                before_lhs="2*(x + 1)",
                after_lhs="2*x + 2",
                variable="x",
            ),
        ]
        results = verify_steps(inputs)
        assert len(results) == 2
        assert all(r.is_correct for r in results)

    def test_parse_error_returns_false(self):
        inp = StepVerifyInput(
            step_number=1,
            before_lhs="x@@@bad",
            after_lhs="x",
            variable="x",
        )
        r = verify_step(inp)
        assert not r.is_correct

    def test_step_number_preserved(self):
        inp = StepVerifyInput(
            step_number=5,
            before_lhs="x",
            after_lhs="x",
            variable="x",
        )
        r = verify_step(inp)
        assert r.step_number == 5


# ─────────────────────────────────────────────────────────────────────────────
# kernel_to_plot2d
# ─────────────────────────────────────────────────────────────────────────────

class TestKernelToPlot2D:
    def test_basic_parabola(self):
        req = Plot2DRequest(expression="x**2", x_range=(-5.0, 5.0), num_points=50)
        data = kernel_to_plot2d(req)
        assert isinstance(data, Plot2DData)
        assert len(data.x_values) > 0
        assert len(data.y_values) == len(data.x_values)

    def test_zero_annotation(self):
        req = Plot2DRequest(expression="x**2 - 4", x_range=(-5.0, 5.0), num_points=200, mark_zeros=True)
        data = kernel_to_plot2d(req)
        # Should detect zero crossings near ±2
        assert len(data.annotations) >= 1

    def test_functions_dict_populated(self):
        req = Plot2DRequest(expression="sin(x)")
        data = kernel_to_plot2d(req)
        assert len(data.functions) == 1

    def test_custom_range(self):
        req = Plot2DRequest(expression="x", x_range=(0.0, 10.0))
        data = kernel_to_plot2d(req)
        assert data.x_range == (0.0, 10.0)

    def test_solve_result_wrapper(self):
        sr = SolveResult(solvable=True, answer="zeros: [-2, 2]")
        data = solve_result_to_plot2d(sr, "x**2 - 4", x_range=(-5.0, 5.0))
        assert isinstance(data, Plot2DData)


# ─────────────────────────────────────────────────────────────────────────────
# kernel_to_three
# ─────────────────────────────────────────────────────────────────────────────

class TestKernelToThree:
    def test_basic_surface(self):
        req = Plot3DRequest(
            expression="x**2 + y**2",
            x_range=(-3.0, 3.0),
            y_range=(-3.0, 3.0),
            grid_points=10,
        )
        data = kernel_to_three(req)
        assert isinstance(data, Three3DData)
        assert len(data.x_values) > 0

    def test_z_values_correct(self):
        req = Plot3DRequest(
            expression="x + y",
            x_range=(0.0, 2.0),
            y_range=(0.0, 2.0),
            grid_points=3,
        )
        data = kernel_to_three(req)
        for xv, yv, zv in zip(data.x_values, data.y_values, data.z_values):
            assert abs(zv - (xv + yv)) < 1e-6

    def test_surface_func_stored(self):
        req = Plot3DRequest(expression="x*y", grid_points=5)
        data = kernel_to_three(req)
        assert data.surface_func == "x*y"

    def test_z_range_computed(self):
        req = Plot3DRequest(
            expression="x**2",
            x_range=(-2.0, 2.0),
            y_range=(-2.0, 2.0),
            grid_points=5,
        )
        data = kernel_to_three(req)
        assert data.z_range[0] >= 0
        assert data.z_range[1] <= 4.1


# ─────────────────────────────────────────────────────────────────────────────
# generate_svg_diagram
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSVGDiagram:
    def test_svg_from_plot2d(self):
        xs = [float(i) for i in range(-5, 6)]
        ys = [x**2 for x in xs]
        data = Plot2DData(
            title="Parabola",
            x_values=xs,
            y_values=ys,
            x_range=(-5.0, 5.0),
            y_range=(0.0, 25.0),
        )
        svg = generate_svg_diagram(data)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Parabola" in svg

    def test_svg_contains_polyline(self):
        xs = [float(i) for i in range(10)]
        ys = [float(i) for i in range(10)]
        data = Plot2DData(x_values=xs, y_values=ys, x_range=(0.0, 9.0), y_range=(0.0, 9.0))
        svg = generate_svg_diagram(data)
        assert "polyline" in svg

    def test_svg_from_three(self):
        req = Plot3DRequest(expression="x**2 + y**2", grid_points=5)
        three_data = kernel_to_three(req)
        svg = generate_svg_diagram(three_data)
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_svg_invalid_type(self):
        with pytest.raises(TypeError):
            generate_svg_diagram("not a data object")  # type: ignore

    def test_custom_config(self):
        data = Plot2DData(x_values=[0.0, 1.0], y_values=[0.0, 1.0], x_range=(0.0, 1.0), y_range=(0.0, 1.0))
        cfg = SVGConfig(width=800, height=600, stroke_color="#FF0000")
        svg = generate_svg_from_plot2d(data, cfg)
        assert 'width="800"' in svg
        assert "#FF0000" in svg

    def test_svg_annotations(self):
        data = Plot2DData(
            x_values=[-2.0, 0.0, 2.0],
            y_values=[0.0, -4.0, 0.0],
            x_range=(-3.0, 3.0),
            y_range=(-5.0, 5.0),
            annotations=[(-2.0, 0.0, "x=-2"), (2.0, 0.0, "x=2")],
        )
        svg = generate_svg_diagram(data)
        assert "x=-2" in svg
        assert "x=2" in svg


# ─────────────────────────────────────────────────────────────────────────────
# SocraticTurnResult (from types)
# ─────────────────────────────────────────────────────────────────────────────

class TestSocraticTurnResult:
    def test_basic_construction(self):
        r = SocraticTurnResult(text="这道题你再想想，思路是什么？")
        assert r.text.startswith("这道题")
        assert r.step_check_triggered is False

    def test_with_step_check(self):
        r = SocraticTurnResult(text="请检查第2步", step_check_triggered=True)
        assert r.step_check_triggered is True
