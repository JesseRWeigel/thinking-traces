"""Build docs/index.html from results/summary.json.

The page is static HTML with the numbers baked in at build time. No script runs
in the browser, so there is no way for the page to render as an empty shell while
the tests stay green. `scripts/verify.sh` rebuilds the page and fails if the file
in git differs, and separately greps the built page for numbers it recomputes
from the summary, so a stale page cannot ship.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = Path(os.environ.get("THINKTRACE_SUMMARY") or (ROOT / "results" / "summary.json"))
OUT = Path(os.environ.get("THINKTRACE_PAGE") or (ROOT / "docs" / "index.html"))

REPO_URL = "https://github.com/JesseRWeigel/thinking-traces"
CATALOG_URL = "https://github.com/JesseRWeigel/722-things-to-build"

TYPE_LABEL = {
    "arith": "Multi-step arithmetic",
    "deduct": "Logical deduction",
    "recall": "Factual recall",
    "instruct": "Instruction following",
    "overthink": "Altered classics",
}
TYPE_BLURB = {
    "arith": "Three and four step word problems. Serial computation with an exact integer answer.",
    "deduct": "Five runner ordering puzzles with a unique solution, verified by brute force.",
    "recall": "One fact, one short answer. Nothing to compute.",
    "instruct": "Output format constraints checked by code: word counts, key order, forbidden letters.",
    "overthink": "Puzzles that look like famous ones but have a different answer. The literal reading is correct and the memorised answer is wrong.",
}

CSS = """
:root { color-scheme: light dark;
  --bg:#ffffff; --fg:#16181d; --muted:#5b6270; --line:#dfe3ea; --card:#f6f8fb;
  --good:#0d6b3f; --bad:#9a2515; --flat:#5b6270; --accent:#1f4fd8; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1116; --fg:#e6e8ee; --muted:#9aa3b2; --line:#262b36; --card:#161a22;
          --good:#4ade80; --bad:#f87171; --flat:#9aa3b2; --accent:#7aa2ff; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width: 62rem; margin: 0 auto; padding: 2rem 1rem 5rem; }
h1 { font-size: clamp(1.5rem, 4vw, 2.2rem); line-height:1.2; margin:0 0 .4rem; }
h2 { font-size: 1.25rem; margin: 2.4rem 0 .6rem; }
h3 { font-size: 1.02rem; margin: 1.6rem 0 .4rem; }
p, li { color: var(--fg); }
.sub { color: var(--muted); margin:0 0 1.6rem; }
a { color: var(--accent); }
.scroll { overflow-x: auto; border:1px solid var(--line); border-radius:10px; margin:.8rem 0 1.4rem; }
table { border-collapse: collapse; width:100%; min-width: 46rem; font-size:.92rem; }
th, td { padding:.5rem .7rem; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; position:sticky; left:0; background:var(--bg); }
thead th { background:var(--card); font-weight:600; color:var(--muted); text-align:right; }
thead th:first-child { text-align:left; background:var(--card); }
tbody tr:last-child td { border-bottom:none; }
.ci { color: var(--muted); font-size:.85em; }
.helps { color: var(--good); font-weight:600; }
.hurts { color: var(--bad); font-weight:600; }
.indistinguishable { color: var(--flat); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:.9rem 1.1rem; margin:1rem 0; }
.grid { display:grid; gap:.8rem; grid-template-columns:repeat(auto-fit,minmax(14rem,1fr)); }
.k { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }
.v { font-size:1.5rem; font-weight:650; }
code { background:var(--card); padding:.1rem .3rem; border-radius:4px; font-size:.9em; }
footer { color:var(--muted); font-size:.85rem; margin-top:3rem; border-top:1px solid var(--line);
         padding-top:1rem; }
"""


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def signed(x: float) -> str:
    return f"{x * 100:+.1f}"


def verdict_html(cell: dict) -> str:
    v = cell["verdict"]
    words = {
        "helps": "helps",
        "hurts": "hurts",
        "indistinguishable": "indistinguishable",
    }[v]
    return f'<span class="{v}">{words}</span>'


def build(summary: dict) -> str:
    meta = summary["meta"]
    models = meta["models"]
    rows = []
    for model in models:
        cells = [c for c in summary["cells"] if c["model"] == model]
        body = []
        for c in cells:
            p = c["paired"]
            extra = c["extra_s_per_item"]
            per_s = c["acc_points_per_extra_second"]
            per_s_txt = "n/a" if per_s is None else f"{per_s:+.2f}"
            per_k = c["acc_points_per_1k_extra_tokens"]
            per_k_txt = "n/a" if per_k is None else f"{per_k:+.2f}"
            body.append(
                "<tr>"
                f"<td>{html.escape(TYPE_LABEL[c['type']])}</td>"
                f"<td>{pct(c['off']['accuracy'])}<br><span class='ci'>"
                f"[{pct(c['off']['acc_lo'])}, {pct(c['off']['acc_hi'])}]</span></td>"
                f"<td>{pct(c['on']['accuracy'])}<br><span class='ci'>"
                f"[{pct(c['on']['acc_lo'])}, {pct(c['on']['acc_hi'])}]</span></td>"
                f"<td>{signed(p['mean'])}<br><span class='ci'>"
                f"[{signed(p['lo'])}, {signed(p['hi'])}]</span></td>"
                f"<td>{verdict_html(c)}</td>"
                f"<td>{c['off']['gen_tokens_mean']:.0f}</td>"
                f"<td>{c['on']['gen_tokens_mean']:.0f}</td>"
                f"<td>{c['token_ratio']:.1f}x</td>"
                f"<td>{per_k_txt}</td>"
                f"<td>{extra:+.1f}s</td>"
                f"<td>{per_s_txt}</td>"
                "</tr>"
            )
        totals = summary["totals"][model]
        tok_ratio = totals["on"]["gen_tokens"] / max(1, totals["off"]["gen_tokens"])
        acc_off = totals["off"]["correct"] / max(1, totals["off"]["n"])
        acc_on = totals["on"]["correct"] / max(1, totals["on"]["n"])
        rows.append(f"""
<h2>{html.escape(model)}</h2>
<div class="grid">
  <div class="card"><div class="k">Accuracy, thinking off</div><div class="v">{pct(acc_off)}</div>
    <div class="ci">{totals['off']['correct']} of {totals['off']['n']} items</div></div>
  <div class="card"><div class="k">Accuracy, thinking on</div><div class="v">{pct(acc_on)}</div>
    <div class="ci">{totals['on']['correct']} of {totals['on']['n']} items</div></div>
  <div class="card"><div class="k">Generated tokens</div><div class="v">{tok_ratio:.1f}x</div>
    <div class="ci">{totals['off']['gen_tokens']:,} off, {totals['on']['gen_tokens']:,} on</div></div>
  <div class="card"><div class="k">Unusable, thinking on</div>
    <div class="v">{totals['on']['n'] - totals['on']['usable']}</div>
    <div class="ci">{totals['on']['truncated']} hit the {meta['num_predict']} token cap</div></div>
</div>
<div class="scroll"><table>
<thead><tr>
  <th>Task type</th><th>Acc, off</th><th>Acc, on</th><th>Paired diff (pts)</th><th>Verdict</th>
  <th>Tok off</th><th>Tok on</th><th>Token cost</th><th>Pts / 1k extra tok</th>
  <th>Extra decode</th><th>Pts / extra s</th>
</tr></thead>
<tbody>{''.join(body)}</tbody>
</table></div>
""")

    # The headline is derived from the data rather than written by hand, so it
    # cannot drift away from the table underneath it.
    overhead = [c for c in summary["cells"]
                if c["verdict"] == "indistinguishable" and (c["token_ratio"] or 0) >= 2.0]
    helps = [c for c in summary["cells"] if c["verdict"] == "helps"]
    hurts = [c for c in summary["cells"] if c["verdict"] == "hurts"]

    def name_list(cells: list[dict]) -> str:
        if not cells:
            return "none"
        return ", ".join(
            f"{html.escape(c['model'])} {html.escape(TYPE_LABEL[c['type']].lower())}"
            for c in cells
        )

    wasted = sum(c["on"]["gen_tokens_total"] - c["off"]["gen_tokens_total"] for c in overhead)
    waste_line = (
        f" Across those cells the traces burned {wasted:,} extra generated tokens for a"
        " difference this sample size cannot distinguish from zero."
        if overhead else ""
    )
    headline = f"""
<div class="card">
<p><strong>Where the traces paid for themselves:</strong> {name_list(helps)}.</p>
<p><strong>Where they cost at least twice the tokens and returned nothing measurable:</strong>
{name_list(overhead)}.{waste_line}</p>
<p><strong>Where they measurably hurt:</strong> {name_list(hurts)}.</p>
</div>
"""

    flag_rows = []
    for model, ev in summary["flag_check"].items():
        flag_rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{ev['on']['with_trace']} / {ev['on']['n']}</td>"
            f"<td>{ev['off']['with_trace']} / {ev['off']['n']}</td>"
            f"<td>{ev['on']['trace_chars']:,}</td>"
            f"<td>{ev['off']['trace_chars']:,}</td>"
            f"<td>{'yes' if ev['flag_effective'] else 'NO'}</td>"
            "</tr>"
        )

    type_defs = "".join(
        f"<li><strong>{html.escape(TYPE_LABEL[t])}</strong>. {html.escape(TYPE_BLURB[t])}</li>"
        for t in meta["task_types"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>What thinking traces actually buy you, per task type</title>
<meta name="description" content="Thinking on versus thinking off, measured per task type on local models at temperature 0, with intervals and token cost.">
<style>{CSS}</style>
</head>
<body>
<main>
<h1>What thinking traces actually buy you, per task type</h1>
<p class="sub">Thinking on versus thinking off on the same model, same items, temperature
{meta['temperature']}, {meta['n_items_per_type']} items per task type.
Intervals are {html.escape(meta['interval'])}.</p>

<div class="card">
<p><strong>How to read the verdict.</strong> The two conditions see identical items, so the
honest comparison is the paired difference and its interval. When that interval contains
zero the row reads <span class="indistinguishable">indistinguishable</span>, which is a
statement about this sample size and not a claim that the effect is exactly zero. A row that
reads indistinguishable while the token cost column reads several times over is the finding
that matters: the traces were paid for and returned nothing measurable.</p>
</div>
{headline}
{''.join(rows)}

<h2>Did the flag actually take effect</h2>
<p>A <code>think</code> flag that is silently ignored produces two identical conditions and a
null result that means nothing. The flag is therefore checked against the response rather than
the request: a reasoning payload has to be present with it on and absent with it off.</p>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Traces, on</th><th>Traces, off</th><th>Trace chars, on</th>
<th>Trace chars, off</th><th>Flag effective</th></tr></thead>
<tbody>{''.join(flag_rows)}</tbody>
</table></div>

<h2>The task types</h2>
<ul>{type_defs}</ul>

<h2>Method</h2>
<ul>
<li>Local models through the Ollama HTTP API, <code>think</code> passed as a top level field on
<code>/api/chat</code>. The OpenAI compatible endpoint drops that field, which is the trap this
experiment is built to avoid.</li>
<li>Temperature {meta['temperature']}, <code>top_k</code> {meta['top_k']}, seed {meta['seed']},
<code>num_predict</code> {meta['num_predict']}. Prompts are byte identical across the two
conditions. The only difference is the flag.</li>
<li>Grading is closed form code, never a model judging a model. Responses that produced nothing
to grade are counted separately from responses that were graded and found wrong, so a zero in the
accuracy column can never be an empty output in disguise.</li>
<li>Cost is reported two ways. Generated tokens are contention free. Decode seconds come from the
server's own <code>eval_duration</code>, which excludes model load, because the card is shared with
other work and raw wall clock on a shared card measures the neighbours.</li>
<li>Two exchange rates follow from that. Points per 1k extra tokens is the one to trust, since
tokens do not move when a neighbour starts a job. Points per extra second is the number the task
asked for, and it sits next to it with the caveat that the seconds were measured on a contended
card.</li>
</ul>

<footer>
<p>Source, raw cached responses and the verification suite:
<a href="{REPO_URL}">{REPO_URL}</a>.
Part of a public build catalog: <a href="{CATALOG_URL}">722 things to build</a>.</p>
</footer>
</main>
</body>
</html>
"""


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(summary), encoding="utf-8")
    print(f"wrote {OUT.name} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
