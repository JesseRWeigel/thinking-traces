#!/usr/bin/env bash
# Verification for thinking-traces.
#
# Runs offline against the cached raw responses. No GPU, no Ollama, no network.
# Anything it cannot check it fails on rather than skipping, because a skipped
# check reports the same success as one that ran.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SUCCESS_LINE="VERIFY OK: thinking-traces"
FAILED=0
STEP=0

step() { STEP=$((STEP + 1)); printf '\n[%d] %s\n' "$STEP" "$1"; }
ok()   { echo "    ok: $1"; }
bad()  { echo "    FAIL: $1"; FAILED=$((FAILED + 1)); }

# ---------------------------------------------------------------- prerequisites
step "prerequisites"
for tool in python3 node; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    bad "$tool is not installed. Install it and re-run. Without node the independent"
    echo "         recomputation of every accuracy and token cost cannot run, and that"
    echo "         check is the only one that does not share code with the aggregator."
  else
    ok "$tool $("$tool" --version 2>&1 | head -1)"
  fi
done
[ "$FAILED" -eq 0 ] || { echo; echo "$SUCCESS_LINE -- NOT REACHED"; exit 1; }

# --------------------------------------------------------------------- fixtures
step "cached responses are present"
RAW_COUNT=$(cat data/raw/*.jsonl 2>/dev/null | grep -c . || true)
if [ "${RAW_COUNT:-0}" -lt 100 ]; then
  bad "only ${RAW_COUNT:-0} cached responses under data/raw, expected the full run"
else
  ok "$RAW_COUNT cached raw responses across $(ls data/raw/*.jsonl | wc -l) condition files"
fi

step "every cached response still belongs to its prompt"
PYTHONPATH=src python3 - <<'PY' || bad "cached responses do not match the current item set"
import json, pathlib, sys
sys.path.insert(0, "src")
from thinktrace.items import build_items
from thinktrace.runner import prompt_sha

items = build_items()
want = {i["id"]: prompt_sha(i["prompt"]) for i in items}
problems, checked = [], 0
seen = {}
for path in sorted(pathlib.Path("data/raw").glob("*.jsonl")):
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        checked += 1
        seen.setdefault((rec["model"], bool(rec["think_requested"])), set()).add(rec["item_id"])
        if rec["item_id"] not in want:
            problems.append(f"{path.name}:{n} unknown item {rec['item_id']}")
        elif "prompt_sha" not in rec:
            problems.append(f"{path.name}:{n} {rec['item_id']} has no prompt fingerprint")
        elif rec["prompt_sha"] != want[rec["item_id"]]:
            problems.append(f"{path.name}:{n} {rec['item_id']} was collected against a "
                            "different prompt, delete the line and re-run it")
# A partial run must not pass. Every model needs both conditions, and every
# condition needs a response for every item, or the paired comparison is being
# taken over a subset that nobody chose.
models = sorted({m for m, _ in seen})
if len(models) < 2:
    problems.append(f"only {len(models)} model(s) measured, the task calls for at least two")
for model in models:
    for think in (False, True):
        got = seen.get((model, think), set())
        if len(got) != len(want):
            problems.append(f"{model} think={think}: {len(got)} responses for {len(want)} items")
print(f"    checked {checked} responses against the current item set")
print(f"    models: {', '.join(models)}, both conditions, {len(want)} items each")
for p in problems[:8]:
    print(f"    FAIL {p}")
sys.exit(1 if problems else 0)
PY

# ------------------------------------------------------------------- unit suite
step "unit suite"
UNIT_LOG="$TMP/unit.log"
PYTHONPATH=src python3 -m unittest discover -s tests -v >"$UNIT_LOG" 2>&1
UNIT_RC=$?
UNIT_COUNT=$(grep -Eo '^Ran [0-9]+ tests?' "$UNIT_LOG" | grep -Eo '[0-9]+' || echo 0)
if [ "$UNIT_RC" -ne 0 ]; then
  bad "unit suite exited $UNIT_RC"
  tail -25 "$UNIT_LOG" | sed 's/^/      /'
else
  ok "$UNIT_COUNT tests passed"
fi
if [ "${UNIT_COUNT:-0}" -lt 60 ]; then
  bad "only $UNIT_COUNT tests ran, the suite has been gutted"
fi

# ----------------------------------------------------------- item determinism
step "item set is reproducible from its generator"
cp -a src "$TMP/src-items"
( cd "$TMP" && PYTHONPATH="$TMP/src-items" python3 - <<'PY' >"$TMP/items-regen.json"
import json, sys
from thinktrace.items import build_items
json.dump(build_items(), sys.stdout, indent=2, ensure_ascii=False)
sys.stdout.write("\n")
PY
) || bad "item generator crashed"
if diff -q data/items.json "$TMP/items-regen.json" >/dev/null 2>&1; then
  ok "data/items.json matches a fresh generation ($(python3 -c 'import json;print(len(json.load(open("data/items.json"))))') items)"
else
  bad "data/items.json differs from a fresh generation"
  diff data/items.json "$TMP/items-regen.json" | head -10 | sed 's/^/      /'
fi

# --------------------------------------------------- analysis is reproducible
step "summary is reproducible from the cached responses"
THINKTRACE_SUMMARY="$TMP/summary.json" PYTHONPATH=src python3 -m thinktrace.analyze \
  >"$TMP/analyze.log" 2>&1
ANALYZE_RC=$?
if [ "$ANALYZE_RC" -ne 0 ]; then
  bad "analysis exited $ANALYZE_RC"
  tail -15 "$TMP/analyze.log" | sed 's/^/      /'
elif diff -q results/summary.json "$TMP/summary.json" >/dev/null 2>&1; then
  ok "results/summary.json matches a fresh analysis of data/raw"
else
  bad "results/summary.json is stale relative to data/raw"
  diff results/summary.json "$TMP/summary.json" | head -10 | sed 's/^/      /'
fi

# ------------------------------------------------------------- flag really took
step "the think flag demonstrably took effect"
python3 - <<'PY' || bad "think flag evidence check failed"
import json, sys, pathlib
s = json.loads(pathlib.Path("results/summary.json").read_text())
bad = []
for model, ev in s["flag_check"].items():
    on, off = ev["on_trace_fraction"], ev["off_trace_fraction"]
    print(f"    {model}: reasoning payload present on {on*100:.0f}% of think-on responses, "
          f"{off*100:.0f}% of think-off responses")
    if not (on >= 0.9 and off <= 0.02):
        bad.append(model)
if bad:
    print("    FAIL: flag never took effect for " + ", ".join(bad))
    sys.exit(1)
PY

# ------------------------------------------------------------- noise floor
step "self-disagreement floor has been measured"
python3 - <<'PY' || bad "noise floor not measured, so no effect size can be interpreted"
import json, pathlib, sys
s = json.loads(pathlib.Path("results/summary.json").read_text())
floor = s["noise_floor"]
if not floor.get("measured"):
    print(f"    COULD NOT CHECK: {floor.get('reason')}")
    print("    Temperature 0 is not determinism on this backend, so without a replicate")
    print("    run there is no floor to compare an effect against. Fix with:")
    print("      python3 -m thinktrace.runner --model <m> --think both --tag rep2 --per-type 8")
    sys.exit(1)
for key, cell in sorted(floor["by_condition"].items()):
    print(f"    {key}: {cell['grade_flips']}/{cell['n']} grades flipped on an unchanged "
          f"re-run ({cell['grade_flip_rate']*100:.1f}%, "
          f"95% CI {cell['rate_lo']*100:.1f} to {cell['rate_hi']*100:.1f})")
print(f"    floor taken as the widest upper bound: {floor['floor_upper_bound']*100:.1f} points")
if any(c["clears_noise_floor"] is None for c in s["cells"]):
    print("    FAIL: some cells carry no floor comparison")
    sys.exit(1)
PY

# ------------------------------------------------------- independent checker
step "independent recomputation (node, shares no code with the analysis)"
node tools/independent_check.js >"$TMP/indep.log" 2>&1
INDEP_RC=$?
if [ "$INDEP_RC" -ne 0 ]; then
  bad "independent check exited $INDEP_RC"
  tail -20 "$TMP/indep.log" | sed 's/^/      /'
else
  sed 's/^/    /' "$TMP/indep.log"
fi

# ------------------------------------------------------------------- the page
step "published page carries the measured numbers"
THINKTRACE_PAGE="$TMP/index.html" PYTHONPATH=src python3 -m thinktrace.site >/dev/null 2>&1 \
  || bad "page builder crashed"
if diff -q docs/index.html "$TMP/index.html" >/dev/null 2>&1; then
  ok "docs/index.html matches a fresh build"
else
  bad "docs/index.html is stale, rebuild with python3 -m thinktrace.site"
fi
python3 tools/check_page.py docs/index.html results/summary.json 2>&1 | sed 's/^/    /'
[ "${PIPESTATUS[0]}" -eq 0 ] || bad "page content check failed"

# ------------------------------------------------------------------- hygiene
step "no private paths, no credential-shaped strings, no binary blind spots"
python3 - <<'PY' || bad "hygiene scan failed"
import re, sys, pathlib

root = pathlib.Path(".")
skip_dirs = {".git", "node_modules", "__pycache__"}
# The scan reads bytes, not text. One NUL byte makes git and grep classify a file
# as binary and skip it silently, which is how a real token got committed here
# once and reported clean.
# Each pattern is assembled from fragments so the literal string it hunts for
# never appears in this file. A scanner that trips on its own source teaches
# people to ignore it, and the first thing they ignore is a real hit.
patterns = {
    "home path": re.compile(b"/" + b"home/[a-z]"),
    "tilde project path": re.compile(b"~/" + b"Projects"),
    "private workspace name": re.compile(b"thou" + b"sand", re.IGNORECASE),
    "github token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{16,}"),
    "openai key": re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    "aws key id": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key block": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
problems, scanned, nul_files = [], 0, []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    if any(part in skip_dirs for part in path.parts):
        continue
    data = path.read_bytes()
    scanned += 1
    if b"\x00" in data:
        nul_files.append(str(path))
    for label, rx in patterns.items():
        for m in rx.finditer(data):
            problems.append(f"{path}: {label} -> {m.group(0)[:24]!r}")
if nul_files:
    problems.append("NUL byte in " + ", ".join(nul_files) +
                    " (write it as the escape \\0 so text tools can still read the file)")
print(f"    scanned {scanned} files as raw bytes")
for p in problems[:10]:
    print(f"    FAIL {p}")
sys.exit(1 if problems else 0)
PY

# ------------------------------------------------------------------- readme
step "README reflects this run"
README=README.md
README_BAD=0
if [ ! -f "$README" ]; then
  bad "no README.md"; README_BAD=1
else
  grep -q '^## Status' "$README" || { bad "README has no Status section"; README_BAD=1; }
  # Whole line match. A substring match would be satisfied by a pasted
  # "VERIFY OK: thinking-traces -- NOT REACHED", which is the failure line,
  # so a failing run could document itself as a passing one.
  grep -qxF "$SUCCESS_LINE" "$README" \
    || { bad "README Status has no line that is exactly the success line"; README_BAD=1; }
  # Matches the exact phrasing this script prints, so the count in the README can
  # only be right if it was pasted from a real run. A pasted count goes stale the
  # moment someone adds a test, and this is what makes that a failure.
  grep -qF "$UNIT_COUNT tests passed" "$README" \
    || { bad "README quotes a different test count than the $UNIT_COUNT that just ran"; README_BAD=1; }
  if grep -q 'TODO' "$README"; then bad "README still contains TODO"; README_BAD=1; fi
  [ "$README_BAD" -eq 0 ] && ok "Status section present, success line and test count match"
fi

# ------------------------------------------------------------------ sabotage
step "sabotage suite"
bash scripts/sabotage.sh >"$TMP/sab.log" 2>&1
SAB_RC=$?
grep -E '^(--- SABOTAGE|  \(a\)|  \(b\)|  \(c\)|  [0-9]+ proven|SABOTAGE)' "$TMP/sab.log" \
  | sed 's/^/    /'
if [ "$SAB_RC" -ne 0 ]; then
  bad "sabotage suite exited $SAB_RC"
  tail -20 "$TMP/sab.log" | sed 's/^/      /'
fi

# -------------------------------------------------------------------- verdict
echo
if [ "$FAILED" -ne 0 ]; then
  echo "$FAILED check(s) failed"
  echo "$SUCCESS_LINE -- NOT REACHED"
  exit 1
fi
echo "$SUCCESS_LINE"
