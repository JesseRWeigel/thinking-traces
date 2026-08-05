#!/usr/bin/env python3
"""Assert the built page actually carries the measured numbers.

Rebuilding the page and diffing it catches staleness. It does not catch a
template that dropped a column, because the rebuilt page and the committed page
would be equally empty. So this checks the page against the summary directly: for
every model and task type, the accuracy strings, the paired interval and the
token ratio have to appear in the served HTML.

It also refuses `overflow-x: hidden` on body. That declaration hides real overflow
and makes any scrollWidth probe vacuous, so it is the one CSS rule this project
treats as a defect rather than a fix.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    page_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "index.html"
    summary_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "summary.json"
    page = page_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    problems: list[str] = []
    checks = 0

    def want(needle: str, label: str) -> None:
        nonlocal checks
        checks += 1
        if needle not in page:
            problems.append(f"{label}: {needle!r} missing from the page")

    for cell in summary["cells"]:
        tag = f"{cell['model']}/{cell['type']}"
        want(f"{cell['off']['accuracy'] * 100:.1f}%", f"{tag} off accuracy")
        want(f"{cell['on']['accuracy'] * 100:.1f}%", f"{tag} on accuracy")
        want(f"{cell['paired']['mean'] * 100:+.1f}", f"{tag} paired mean")
        want(f"[{cell['paired']['lo'] * 100:+.1f}, {cell['paired']['hi'] * 100:+.1f}]",
             f"{tag} paired interval")
        pu = cell["paired_usable_only"]
        want(f"[{pu['lo'] * 100:+.1f}, {pu['hi'] * 100:+.1f}] n={pu['n']}",
             f"{tag} answered-only interval")
        want(f"{cell['token_ratio']:.1f}x", f"{tag} token ratio")
        want(cell["verdict"], f"{tag} verdict")
        # Usable counts have to be on the page, or a zero accuracy cell is
        # indistinguishable from a cell that produced nothing to grade.
        want(f"{cell['off']['usable']} / {cell['on']['usable']}", f"{tag} usable counts")
        for cond in ("off", "on"):
            v = cell[cond]["accuracy_usable"]
            want("n/a" if v is None else f"{v * 100:.1f}%", f"{tag} {cond} accuracy among usable")
        if cell["budget_limited"]:
            want("budget limited", f"{tag} budget limited marker")

    floor = summary["noise_floor"]
    if floor.get("measured"):
        want(f"{floor['floor_upper_bound'] * 100:.1f} points", "noise floor upper bound")
        for _, cell in sorted(floor["by_condition"].items()):
            want(f"{cell['grade_flips']} / {cell['n']}", "noise floor flip count")
    else:
        want("Not measured", "noise floor not-measured notice")

    for model, ev in summary["flag_check"].items():
        want(model, "model name")
        want(f"{ev['on']['with_trace']} / {ev['on']['n']}", f"{model} trace evidence")

    # A page that renders but shows nothing would still pass a "file exists" test.
    digits = len(re.findall(r"\d", page))
    checks += 1
    if digits < 200:
        problems.append(f"page contains only {digits} digits, it cannot be showing the results")

    checks += 1
    if re.search(r"body\s*\{[^}]*overflow-x\s*:\s*hidden", page, re.IGNORECASE):
        problems.append("body { overflow-x: hidden } hides real overflow, remove it")

    checks += 1
    if "<title>" not in page:
        problems.append("page has no title")

    # Negative control on the checker itself: the same routine run against a page
    # with the numbers stripped out has to complain. Without this, a `want` that
    # silently matched everything would look identical to a passing check.
    stripped = re.sub(r"\d", "", page)
    control_hits = sum(
        1 for cell in summary["cells"]
        if f"{cell['off']['accuracy'] * 100:.1f}%" in stripped
    )
    checks += 1
    if control_hits > 0:
        problems.append("negative control failed: digit-stripped page still matched accuracies")

    for p in problems:
        print(f"  PAGE {p}")
    print(f"page check: {checks} assertions, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
