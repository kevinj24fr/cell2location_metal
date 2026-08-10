"""Tests for the HTML validation report.

The report is generated at the *end* of a run that may have taken an hour. A
KeyError there loses the whole result, so the renderer is tested against partial,
failed and empty inputs rather than only the happy path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from report_html import render_report  # noqa: E402

_FULL = {
    "environment": {
        "platform": "Darwin arm64 (macOS 15.0)",
        "torch_version": "2.13.0",
        "mps_available": True,
        "default_device": "mps",
        "lgamma_mode": "auto",
        "unified_memory_gb": 192.0,
        "fused_nb": {"requested": True, "verified": True, "rejected_reason": None},
    },
    "op_support": {"lgamma": True, "poisson": False},
    "lgamma_parity": [
        {"mode": "contiguous", "case": "broadcast view", "max_abs_error": 1e-6, "max_rel_error": 1e-7, "passed": True},
        {"mode": "stirling", "case": "broadcast view", "max_abs_error": 2e-6, "max_rel_error": 3e-1, "passed": True},
    ],
    "nb_parity": [{"shape": [256, 2000], "max_abs_error": 1e-5, "sum_rel_error": 3e-8, "passed": True}],
    "kernel_benchmarks": [{"benchmark": "abundance matmul", "seconds": {"cpu": 0.09, "mps": 0.012}}],
    "training_benchmarks": [
        {"device": "cpu", "spatial_seconds": 120.0, "final_elbo": 1e5},
        {"device": "mps", "spatial_seconds": 30.0, "final_elbo": 1e5},
    ],
}


def test_renders_a_complete_document():
    html = render_report(_FULL)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")


def test_reports_the_speedup_as_the_hero_number():
    html = render_report(_FULL)
    assert "4.00&times;" in html


def test_verdict_is_positive_when_the_default_path_passes():
    assert "Numerically sound" in render_report(_FULL)


def test_verdict_is_alarming_when_the_default_path_fails():
    """A failure has to read as 'stop', not as a yellow warning among many."""
    results = {**_FULL, "lgamma_parity": [{"mode": "contiguous", "case": "broadcast view", "passed": False}]}
    html = render_report(results)
    assert "Do not trust these results" in html
    assert "without reporting anything" in html


def test_failure_suggests_a_working_mode_when_one_exists():
    results = {
        **_FULL,
        "lgamma_parity": [
            {"mode": "contiguous", "case": "broadcast view", "passed": False},
            {"mode": "stirling", "case": "broadcast view", "passed": True},
        ],
    }
    html = render_report(results)
    assert "CELL2LOCATION_MPS_LGAMMA=stirling" in html


def test_failure_says_use_the_cpu_when_nothing_works():
    results = {
        **_FULL,
        "lgamma_parity": [
            {"mode": "contiguous", "case": "broadcast view", "passed": False},
            {"mode": "stirling", "case": "broadcast view", "passed": False},
        ],
    }
    html = render_report(results)
    assert "accelerator=" in html and "cpu" in html


def test_default_mode_rows_are_marked():
    """Twenty near-identical rows are useless unless the one that runs is obvious."""
    html = render_report(_FULL)
    assert 'class="is-default"' in html
    assert ">default<" in html


def test_survives_an_empty_result_set():
    html = render_report({})
    assert "<!DOCTYPE html>" in html
    assert "Not measured" in html


def test_survives_missing_sections():
    html = render_report({"environment": {"platform": "Darwin arm64", "mps_available": False}})
    assert "unavailable" in html


def test_survives_a_failed_benchmark_entry():
    results = {**_FULL, "training_benchmarks": [{"device": "mps", "error": "RuntimeError: out of memory"}]}
    html = render_report(results)
    assert "<!DOCTYPE html>" in html
    assert "&times;</p>" not in html, "no speedup should be claimed when a device failed"


def test_missing_measurements_render_as_a_dash_not_a_blank():
    results = {**_FULL, "nb_parity": [{"shape": [8, 8], "passed": False}]}
    html = render_report(results)
    assert "&mdash;" in html


def test_escapes_hostile_content():
    """Rejection reasons are exception strings and can contain anything."""
    results = {
        "environment": {
            "platform": "<script>alert('x')</script>",
            "mps_available": True,
            "fused_nb": {"requested": True, "verified": False, "rejected_reason": "<img onerror=1>"},
        }
    }
    html = render_report(results)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "<img onerror" not in html


def test_report_is_self_contained():
    """It has to survive being attached to an issue or opened offline."""
    html = render_report(_FULL)
    for external in ("http://", "https://", "<script", "<link"):
        assert external not in html, f"report should not reference {external}"


def test_dark_mode_is_defined():
    html = render_report(_FULL)
    assert "prefers-color-scheme: dark" in html


def test_status_is_never_colour_alone():
    """Every status chip carries a word next to its dot, for CVD and for print."""
    html = render_report(_FULL)
    assert "match" in html and "native" in html
    assert html.count('class="dot"') >= 3


@pytest.mark.parametrize("missing", ["environment", "op_support", "lgamma_parity", "kernel_benchmarks"])
def test_survives_each_section_individually_absent(missing):
    results = {k: v for k, v in _FULL.items() if k != missing}
    assert "<!DOCTYPE html>" in render_report(results)
