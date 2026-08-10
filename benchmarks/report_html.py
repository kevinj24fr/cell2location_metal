"""Render the Apple-silicon validation results as a self-contained HTML report.

Terminal output scrolls away and does not travel well. This produces one file that
can be attached to an issue, kept next to a paper's methods, or diffed against the
same machine six months and three PyTorch releases later.

No dependencies, no network, no build step -- everything is inlined, so the file
works from a USB stick.
"""

from typing import Any, Dict, List, Optional

__all__ = ["render_report"]

# Palette: categorical slots 1 and 2 from the reference data-viz palette, validated
# for CVD separation and surface contrast in both light and dark mode. Two series is
# the whole chart, so slot order is fixed and never cycled.
_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --plane: #f9f9f7;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --cpu: #2a78d6;
  --gpu: #eb6834;
  --good: #0ca30c;
  --warning: #fab219;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: #1a1a19;
    --plane: #0d0d0d;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --cpu: #3987e5;
    --gpu: #d95926;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 40px 24px 80px;
  background: var(--plane);
  color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-size: 24px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 16px; letter-spacing: 0.02em;
     text-transform: uppercase; color: var(--ink-2); }
.sub { color: var(--ink-2); margin: 0 0 32px; font-size: 14px; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 24px; margin-bottom: 20px;
}
.hero { font-size: 52px; font-weight: 600; line-height: 1.05; margin: 0; }
.hero-label { color: var(--ink-2); font-size: 14px; margin: 6px 0 0; }
.row { display: flex; align-items: baseline; gap: 10px; padding: 7px 0; }
.row + .row { border-top: 1px solid var(--grid); }
.row .k { color: var(--ink-2); min-width: 190px; font-size: 14px; }
.row .v { font-variant-numeric: tabular-nums; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.good .dot { background: var(--good); }
.warning .dot { background: var(--warning); }
.critical .dot { background: var(--critical); }
.good { color: var(--good); }
.warning { color: var(--ink); }
.critical { color: var(--critical); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th {
  text-align: left; font-weight: 600; color: var(--ink-2); font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.03em; padding: 0 10px 8px 0;
  border-bottom: 1px solid var(--axis);
}
td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
td:first-child, th:first-child { padding-left: 0; }
.legend { display: flex; gap: 18px; margin-bottom: 18px; font-size: 13px; color: var(--ink-2); }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block;
          margin-right: 6px; vertical-align: -1px; }
.bar-group { margin-bottom: 20px; }
.bar-title { font-size: 14px; margin-bottom: 8px; }
.bar-line { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; position: relative; }
.bar {
  height: 15px; border-radius: 0 4px 4px 0; min-width: 2px;
  transition: filter 120ms ease;
}
.bar:hover { filter: brightness(1.12); }
.bar-line:hover .tip { opacity: 1; }
.bar-value { font-size: 13px; color: var(--ink-2); font-variant-numeric: tabular-nums; white-space: nowrap; }
.tip {
  position: absolute; right: 0; bottom: 100%; opacity: 0; pointer-events: none;
  background: var(--ink); color: var(--surface); font-size: 12px; padding: 4px 8px;
  border-radius: 5px; white-space: nowrap; transition: opacity 120ms ease; z-index: 2;
}
.note { color: var(--ink-2); font-size: 13px; margin: 14px 0 0; }
.note code { background: var(--plane); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
.tag {
  font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--ink-2); border: 1px solid var(--axis); border-radius: 4px; padding: 1px 4px;
  vertical-align: 1px;
}
tr.is-default td { background: color-mix(in srgb, var(--cpu) 5%, transparent); }
.muted-col { color: var(--muted); }
.note em { font-style: normal; font-weight: 600; }
footer { color: var(--muted); font-size: 12px; text-align: center; margin-top: 32px; }
"""


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _number(value: Optional[float]) -> str:
    """Scientific notation, or an em dash when the measurement was not taken.

    A blank cell reads as zero; an em dash reads as "no data", which is the truth
    when a case raised before it could be measured.
    """
    return f"{value:.2e}" if value is not None else "&mdash;"


def _chip(state: str, label: str) -> str:
    """A status indicator is never colour alone -- always a dot plus a word."""
    return f'<span class="chip {state}"><span class="dot"></span>{_escape(label)}</span>'


def _rows(pairs: List[tuple]) -> str:
    return "".join(
        f'<div class="row"><span class="k">{_escape(k)}</span><span class="v">{v}</span></div>' for k, v in pairs
    )


def _environment_card(env: Dict[str, Any]) -> str:
    available = env.get("mps_available")
    pairs = [
        ("Platform", _escape(env.get("platform", "unknown"))),
        ("PyTorch", _escape(env.get("torch_version", "unknown"))),
        ("Metal backend", _chip("good", "available") if available else _chip("critical", "unavailable")),
        ("Default device", _escape(env.get("default_device", "?"))),
        ("lgamma mode", _escape(env.get("lgamma_mode", "?"))),
    ]
    if env.get("unified_memory_gb"):
        pairs.append(("Unified memory", f"{env['unified_memory_gb']} GB"))

    fused = env.get("fused_nb") or {}
    if fused.get("requested"):
        if fused.get("verified"):
            state = _chip("good", "verified and active")
        elif fused.get("verified") is False:
            state = _chip("critical", f"rejected — {_escape(fused.get('rejected_reason', 'unknown'))}")
        else:
            state = _chip("warning", "requested, not yet verified")
        pairs.append(("Fused NB kernel", state))

    return f'<div class="card"><h2>Environment</h2>{_rows(pairs)}</div>'


def _verdict_card(results: Dict[str, Any]) -> str:
    parity = list(results.get("lgamma_parity", [])) + list(results.get("nb_parity", []))
    default_path = [r for r in parity if r.get("mode", "contiguous") == "contiguous"]
    failures = [r for r in default_path if not r.get("passed")]

    if not default_path:
        headline, state, detail = "Not measured", "warning", "Metal was unavailable, so no parity check ran."
    elif failures:
        headline, state = "Do not trust these results", "critical"
        detail = (
            f"{len(failures)} of {len(default_path)} checks failed on the default numerical path. "
            "The likelihood disagrees with the CPU by more than rounding explains, which means a "
            "training run would descend a wrong objective without reporting anything."
        )
    else:
        headline, state = "Numerically sound", "good"
        detail = (
            f"All {len(default_path)} checks on the default numerical path agree with the CPU "
            f"({len(parity)} were run in total, across every fallback mode). "
            "The negative-binomial likelihood computes the same thing on Metal as it does on CPU."
        )

    training = [r for r in results.get("training_benchmarks", []) if "error" not in r]
    by_device = {r["device"]: r for r in training}
    hero = ""
    if "cpu" in by_device and "mps" in by_device:
        speedup = by_device["cpu"]["spatial_seconds"] / by_device["mps"]["spatial_seconds"]
        hero = (
            f'<p class="hero">{speedup:.2f}&times;</p>'
            f'<p class="hero-label">spatial model, Metal vs CPU, on synthetic data</p>'
        )

    return (
        f'<div class="card"><h2>Verdict</h2>'
        f'<div style="margin-bottom:14px">{_chip(state, headline)}</div>'
        f"{hero}"
        f'<p class="note">{detail}</p></div>'
    )


def _op_support_card(ops: Dict[str, bool]) -> str:
    if not ops:
        return ""
    body = "".join(
        f"<tr><td>{_escape(name)}</td><td>"
        + (_chip("good", "native") if ok else _chip("warning", "CPU fallback"))
        + "</td></tr>"
        for name, ok in ops.items()
    )
    return (
        '<div class="card"><h2>Operator coverage</h2>'
        f"<table><thead><tr><th>Operation</th><th>Status</th></tr></thead><tbody>{body}</tbody></table>"
        '<p class="note">A fallback is correct, just slower — the op runs on the CPU and the '
        "result comes back. Nothing here affects the numbers.</p></div>"
    )


def _parity_card(results: Dict[str, Any]) -> str:
    rows = results.get("lgamma_parity", [])
    if not rows:
        return ""

    body = ""
    for row in rows:
        status = _chip("good", "match") if row.get("passed") else _chip("critical", "mismatch")
        abs_err = row.get("max_abs_error")
        rel_err = row.get("max_rel_error")
        # The default path is what almost every user will actually run; the other modes
        # are here so a failure has a documented way out.
        is_default = row.get("mode") == "contiguous"
        mode_cell = _escape(row.get("mode"))
        row_class = ""
        if is_default:
            mode_cell += ' <span class="tag">default</span>'
            row_class = ' class="is-default"'
        body += (
            f"<tr{row_class}>"
            f"<td>{mode_cell}</td>"
            f"<td>{_escape(row.get('case'))}</td>"
            f"<td>{_number(abs_err)}</td>"
            f'<td class="muted-col">{_number(rel_err)}</td>'
            f"<td>{status}</td></tr>"
        )

    return (
        '<div class="card"><h2>lgamma parity vs CPU</h2>'
        "<table><thead><tr><th>Mode</th><th>Input</th><th>Max abs error</th>"
        "<th>Max rel error*</th><th></th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        '<p class="note">The <strong>broadcast view</strong> row is the one that matters: it is the '
        "shape the negative-binomial likelihood actually produces, and the one PyTorch has "
        "miscomputed on this backend.</p>"
        '<p class="note">*Relative error is shown for reference only and is <em>not</em> the pass '
        "criterion. lgamma passes through zero at x=1 and x=2, so relative error there is unbounded "
        "for any correct implementation — a large number in that column beside a green “match” is "
        "expected, not a contradiction. The verdict uses a combined absolute/relative budget, and "
        "what reaches a summed log-likelihood is the absolute error.</p></div>"
    )


def _nb_parity_card(rows: List[Dict[str, Any]]) -> str:
    """The likelihood itself, at full scale. This is the number that decides whether a
    training run can be believed."""
    if not rows:
        return ""

    body = ""
    for row in rows:
        status = _chip("good", "match") if row.get("passed") else _chip("critical", "mismatch")
        shape = " &times; ".join(str(int(d)) for d in (row.get("shape") or []))
        body += (
            "<tr>"
            f"<td>{shape}</td>"
            f"<td>{_number(row.get('max_abs_error'))}</td>"
            f"<td>{_number(row.get('sum_rel_error'))}</td>"
            f"<td>{status}</td></tr>"
        )

    return (
        '<div class="card"><h2>Negative-binomial likelihood parity</h2>'
        "<table><thead><tr><th>Shape</th><th>Max elementwise error</th>"
        "<th>Summed rel error</th><th></th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        '<p class="note">The summed column is the one training optimises. Elementwise noise can '
        "cancel across a sum or accumulate in it, and only this tells you which happened.</p></div>"
    )


def _benchmark_card(rows: List[Dict[str, Any]]) -> str:
    """Grouped horizontal bars. Two series, fixed colours, every bar directly labelled."""
    if not rows:
        return ""

    timings = [
        (row["benchmark"], row["seconds"].get("cpu"), row["seconds"].get("mps"))
        for row in rows
        if isinstance(row.get("seconds"), dict)
    ]
    timings = [t for t in timings if t[1] or t[2]]
    if not timings:
        return ""

    longest = max(v for _, cpu, mps in timings for v in (cpu, mps) if v)

    groups = ""
    for label, cpu, mps in timings:
        lines = ""
        for series, value, colour in (("CPU", cpu, "var(--cpu)"), ("Metal", mps, "var(--gpu)")):
            if value is None:
                continue
            width = max(1.0, 100.0 * value / longest)
            lines += (
                f'<div class="bar-line">'
                f'<span class="bar" style="width:{width:.1f}%;background:{colour}"></span>'
                f'<span class="bar-value">{value * 1e3:.1f} ms</span>'
                f'<span class="tip">{_escape(series)}: {value * 1e3:.2f} ms</span>'
                f"</div>"
            )
        speedup = ""
        if cpu and mps:
            speedup = f" &middot; {cpu / mps:.2f}&times;"
        groups += f'<div class="bar-group"><div class="bar-title">{_escape(label)}{speedup}</div>{lines}</div>'

    # Only legend the series that actually produced bars -- a legend entry with nothing
    # on the chart reads as missing data rather than as "not measured".
    entries = []
    if any(cpu for _, cpu, _ in timings):
        entries.append('<span><span class="swatch" style="background:var(--cpu)"></span>CPU</span>')
    if any(mps for _, _, mps in timings):
        entries.append('<span><span class="swatch" style="background:var(--gpu)"></span>Metal</span>')
    legend = f'<div class="legend">{"".join(entries)}</div>' if len(entries) > 1 else ""
    return (
        f'<div class="card"><h2>Kernel benchmarks</h2>{legend}{groups}'
        '<p class="note">Shapes mirror a Visium slide against a reference signature matrix. '
        "Shorter is better.</p></div>"
    )


def _guidance_card(results: Dict[str, Any]) -> str:
    parity = list(results.get("lgamma_parity", []))
    default_failed = any(r.get("mode") == "contiguous" and not r.get("passed") for r in parity)
    if not default_failed:
        return ""

    working = None
    by_mode: Dict[str, List[bool]] = {}
    for row in parity:
        by_mode.setdefault(row["mode"], []).append(bool(row.get("passed")))
    for mode, passes in by_mode.items():
        if mode not in ("contiguous", "native") and all(passes):
            working = mode
            break

    if working:
        action = (
            f"<p>Set <code>CELL2LOCATION_MPS_LGAMMA={_escape(working)}</code> before training. "
            "That mode passed every check on this machine.</p>"
        )
    else:
        action = (
            '<p>No lgamma mode passed. Train with <code>accelerator="cpu"</code> and please '
            "open an issue with this report attached.</p>"
        )

    return f'<div class="card"><h2>What to do</h2>{action}</div>'


def render_report(results: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    """Build the complete HTML document from a results dictionary."""
    env = results.get("environment", {})
    stamp = f"Generated {_escape(generated_at)}" if generated_at else "cell2location Apple silicon validation"

    sections = [
        _verdict_card(results),
        _guidance_card(results),
        _environment_card(env),
        _parity_card(results),
        _nb_parity_card(results.get("nb_parity", [])),
        _benchmark_card(results.get("kernel_benchmarks", [])),
        _op_support_card(results.get("op_support", {})),
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cell2location on Apple silicon</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>cell2location on Apple silicon</h1>
  <p class="sub">{_escape(env.get('platform', 'unknown platform'))} &middot; PyTorch {_escape(env.get('torch_version', '?'))}</p>
  {''.join(s for s in sections if s)}
  <footer>{stamp}</footer>
</div>
</body>
</html>
"""
