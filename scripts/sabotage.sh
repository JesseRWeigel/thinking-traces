#!/usr/bin/env bash
# Sabotage suite.
#
# For each core computation: break it in a COPY of the tree, then prove three
# separate things in order.
#
#   (a) the patch APPLIED          the file on disk actually changed
#   (b) it CHANGED the output      the artefact regenerated from the sabotaged
#                                  code differs from the committed one
#   (c) a check CAUGHT it          the named detector exits nonzero
#
# A sabotage that fails (a) or (b) is a no-op, and this script reports it as a
# no-op and fails rather than crediting the detector. That failure mode is the
# reason this file exists: an attack that changed nothing produces a confident
# write up saying the verify has a gap, and the gap is in the attack.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

banner() { printf '\n--- %s\n' "$1"; }

# run_sabotage <name> <target-file> <python-patch-expr> <artifact> <detector...>
#   artifact is "summary" or "page"
run_sabotage() {
  local name="$1" target="$2" patch="$3" artifact="$4"
  shift 4
  local detector=("$@")
  local dir="$WORK/$name"

  banner "SABOTAGE $name"
  rm -rf "$dir"
  cp -a "$ROOT" "$dir"
  rm -rf "$dir/.git"

  # (a) prove the patch applied
  python3 - "$dir/$target" "$patch" <<'PY'
import sys, pathlib
path, expr = pathlib.Path(sys.argv[1]), sys.argv[2]
old, new = expr.split("||>>")
text = path.read_text(encoding="utf-8")
if old not in text:
    print(f"  ANCHOR NOT FOUND in {path.name}: {old[:70]!r}")
    sys.exit(3)
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
  if [ $? -ne 0 ]; then
    echo "  (a) FAILED: patch anchor missing, the sabotage is a no-op"
    FAIL=$((FAIL + 1)); return
  fi
  if diff -q "$ROOT/$target" "$dir/$target" >/dev/null 2>&1; then
    echo "  (a) FAILED: $target is byte identical after patching, the sabotage is a no-op"
    FAIL=$((FAIL + 1)); return
  fi
  echo "  (a) patch applied: $target differs from the original"

  # (b) prove the measured output changed
  local changed=0
  if [ "$artifact" = "summary" ]; then
    ( cd "$dir" && THINKTRACE_SUMMARY="$dir/sabotaged.json" \
        PYTHONPATH=src python3 -m thinktrace.analyze >/dev/null 2>&1 )
    if [ ! -f "$dir/sabotaged.json" ]; then
      echo "  (b) sabotaged analysis produced no summary at all, which is itself a change"
      changed=1
    elif ! diff -q "$ROOT/results/summary.json" "$dir/sabotaged.json" >/dev/null 2>&1; then
      changed=1
      echo "  (b) measured output changed: $(diff <(python3 -c '
import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=1,sort_keys=True))
' "$ROOT/results/summary.json") <(python3 -c '
import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=1,sort_keys=True))
' "$dir/sabotaged.json") | grep -c '^[<>]') differing summary lines"
    fi
  else
    ( cd "$dir" && THINKTRACE_PAGE="$dir/sabotaged.html" \
        PYTHONPATH=src python3 -m thinktrace.site >/dev/null 2>&1 )
    if [ ! -f "$dir/sabotaged.html" ]; then
      echo "  (b) sabotaged builder produced no page at all, which is itself a change"
      changed=1
    elif ! diff -q "$ROOT/docs/index.html" "$dir/sabotaged.html" >/dev/null 2>&1; then
      changed=1
      echo "  (b) measured output changed: $(diff "$ROOT/docs/index.html" "$dir/sabotaged.html" | grep -c '^[<>]') differing page lines"
    fi
  fi
  if [ "$changed" -ne 1 ]; then
    echo "  (b) FAILED: the artefact is unchanged, so this attack proves nothing about the checks"
    FAIL=$((FAIL + 1)); return
  fi

  # (c) prove a check catches it
  ( cd "$dir" && PYTHONPATH=src "${detector[@]}" ) >"$WORK/$name.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  (c) FAILED: detector '${detector[*]}' still exits 0 on sabotaged code"
    sed -n '1,15p' "$WORK/$name.log" | sed 's/^/      /'
    FAIL=$((FAIL + 1)); return
  fi
  echo "  (c) caught by '${detector[*]}' (exit $rc):"
  grep -Ei 'mismatch|FAIL|AssertionError|PAGE ' "$WORK/$name.log" | head -3 | sed 's/^/      /'
  PASS=$((PASS + 1))
}

echo "sabotage suite, working in $WORK"

# 1. Grading: every numeric answer scored correct.
run_sabotage numeric-grader-always-true src/thinktrace/grade.py \
'def grade_numeric(item: dict, pred: str) -> bool:
    norm = normalize_text(pred)||>>def grade_numeric(item: dict, pred: str) -> bool:
    return True
    norm = normalize_text(pred)' \
  summary node tools/independent_check.js

# 2. Usability: empty output counted as a graded response.
run_sabotage empty-output-counted-usable src/thinktrace/grade.py \
'    usable = bool(body) and bool(pred)||>>    usable = True' \
  summary python3 -m unittest discover -s tests

# 3. Interval: Wilson collapsed to zero width, which makes every accuracy look exact.
run_sabotage wilson-zero-width src/thinktrace/stats.py \
'    p = k / n
    denom = 1.0 + z * z / n||>>    p = k / n
    return (p, p)
    denom = 1.0 + z * z / n' \
  summary python3 -m unittest discover -s tests

# 4. Paired statistic: sign of the effect flipped, so "hurts" reads as "helps".
run_sabotage paired-diff-sign-flip src/thinktrace/stats.py \
'    mean, sd = mean_sd(diffs)||>>    mean, sd = mean_sd(diffs)
    mean = -mean' \
  summary node tools/independent_check.js

# 5. Cost: thinking tokens dropped from the token aggregate.
run_sabotage token-cost-undercount src/thinktrace/analyze.py \
'            "gen_tokens": rec.get("eval_count", 0),||>>            "gen_tokens": rec.get("prompt_eval_count", 0),' \
  summary node tools/independent_check.js

# 6. Flag evidence: trace detection reads the wrong field, so a silently ignored
#    think flag would look like it took effect.
run_sabotage flag-evidence-wrong-field src/thinktrace/analyze.py \
'        if (rec.get("thinking") or "").strip():||>>        if (rec.get("content") or "").strip():' \
  summary node tools/independent_check.js

# 7. Ground truth: one arithmetic answer shifted by one.
run_sabotage ground-truth-off-by-one src/thinktrace/items.py \
'                    "answer": str(value),||>>                    "answer": str(value + 1),' \
  summary python3 -m unittest discover -s tests

# 8. Publication: the page built with the numbers blanked out.
run_sabotage page-numbers-blanked src/thinktrace/site.py \
'def pct(x: float) -> str:
    return f"{x * 100:.1f}%"||>>def pct(x: float) -> str:
    return "n/a"' \
  page python3 tools/check_page.py sabotaged.html

banner "sabotage summary"
echo "  $PASS proven attacks caught, $FAIL failed"
if [ "$FAIL" -ne 0 ]; then
  echo "SABOTAGE SUITE FAILED"
  exit 1
fi
echo "SABOTAGE SUITE PASSED"
