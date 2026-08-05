"""Aggregate cached raw responses into results/summary.json.

Reads only data/raw/*.jsonl and data/items.json, so it runs on any machine with
no GPU and no Ollama.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .grade import grade_response
from .items import TASK_TYPES, build_items
from .stats import paired_diff, ratio_ci, wilson

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
# Overridable so verify can regenerate into a scratch path and diff, rather than
# writing over the committed result and then comparing it against itself.
OUT = Path(os.environ.get("THINKTRACE_SUMMARY") or (ROOT / "results" / "summary.json"))


def load_raw() -> list[dict]:
    records = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def thinking_evidence(records: list[dict]) -> dict:
    """Did the `think` flag actually take effect, per model?

    The flag is checked against the response, not against the request. A model
    whose two conditions both produce empty `thinking` payloads was never
    switched, and its numbers are worthless.
    """
    out: dict[str, dict] = {}
    for rec in records:
        m = rec["model"]
        cond = "on" if rec["think_requested"] else "off"
        cell = out.setdefault(m, {
            "on": {"n": 0, "with_trace": 0, "trace_chars": 0},
            "off": {"n": 0, "with_trace": 0, "trace_chars": 0},
        })[cond]
        cell["n"] += 1
        if (rec.get("thinking") or "").strip():
            cell["with_trace"] += 1
        cell["trace_chars"] += len(rec.get("thinking") or "")
    for m, cell in out.items():
        on, off = cell["on"], cell["off"]
        on_frac = on["with_trace"] / on["n"] if on["n"] else 0.0
        off_frac = off["with_trace"] / off["n"] if off["n"] else 0.0
        cell["on_trace_fraction"] = on_frac
        cell["off_trace_fraction"] = off_frac
        # The flag took effect if traces are the norm with it on and absent with
        # it off. Both thresholds matter: a model that always emits traces was
        # never switched off, and one that never emits them was never switched on.
        cell["flag_effective"] = on_frac >= 0.9 and off_frac <= 0.02
    return out


def analyze() -> dict:
    items = {it["id"]: it for it in build_items()}
    records = load_raw()
    if not records:
        raise SystemExit("no cached responses under data/raw")

    graded: list[dict] = []
    for rec in records:
        item = items[rec["item_id"]]
        g = grade_response(item, rec.get("content", ""), rec.get("done_reason", ""))
        graded.append({
            "model": rec["model"],
            "cond": "on" if rec["think_requested"] else "off",
            "type": rec["type"],
            "item_id": rec["item_id"],
            "usable": g["usable"],
            "correct": g["correct"],
            "pred": g["pred"],
            "done_reason": rec.get("done_reason", ""),
            "gen_tokens": rec.get("eval_count", 0),
            "prompt_tokens": rec.get("prompt_eval_count", 0),
            "wall_s": rec.get("wall_s", 0.0),
            # Decode time only. The box is shared, so a model can be evicted and
            # reloaded mid run; load_duration lands in wall_s and not in gen_s,
            # which makes gen_s the honest per-item cost and wall_s the number a
            # caller would actually experience on a contended machine.
            "gen_s": rec.get("eval_duration_ns", 0) / 1e9,
            "load_s": rec.get("load_duration_ns", 0) / 1e9,
            "thinking_chars": len(rec.get("thinking") or ""),
        })

    models = sorted({g["model"] for g in graded})
    evidence = thinking_evidence(records)

    cells: list[dict] = []
    # A cell with data for only one condition cannot be compared. Dropping it
    # quietly would let a half finished run publish a table that looks complete,
    # so the omission is recorded and reported rather than swallowed.
    incomplete: list[dict] = []
    for model in models:
        for ttype in TASK_TYPES:
            by_cond = {}
            for cond in ("off", "on"):
                sel = [g for g in graded
                       if g["model"] == model and g["type"] == ttype and g["cond"] == cond]
                if not sel:
                    continue
                n = len(sel)
                k = sum(1 for g in sel if g["correct"])
                usable = sum(1 for g in sel if g["usable"])
                truncated = sum(1 for g in sel if g["done_reason"] == "length")
                lo, hi = wilson(k, n)
                by_cond[cond] = {
                    "n": n,
                    "correct": k,
                    "usable": usable,
                    "truncated": truncated,
                    "accuracy": k / n,
                    "acc_lo": lo,
                    "acc_hi": hi,
                    # Accuracy among responses that produced something to grade.
                    # Reported alongside the headline because a cell can score low
                    # for two completely different reasons: the model answered and
                    # was wrong, or the model spent its whole budget reasoning and
                    # never answered. Those call for opposite fixes.
                    "accuracy_usable": (k / usable) if usable else None,
                    "gen_tokens_total": sum(g["gen_tokens"] for g in sel),
                    "gen_tokens_mean": sum(g["gen_tokens"] for g in sel) / n,
                    "wall_s_total": sum(g["wall_s"] for g in sel),
                    "wall_s_mean": sum(g["wall_s"] for g in sel) / n,
                    "gen_s_total": sum(g["gen_s"] for g in sel),
                    "gen_s_mean": sum(g["gen_s"] for g in sel) / n,
                    "load_s_total": sum(g["load_s"] for g in sel),
                }
            if set(by_cond) != {"on", "off"}:
                incomplete.append({
                    "model": model, "type": ttype,
                    "conditions_present": sorted(by_cond),
                })
                continue

            on_by_item = {g["item_id"]: g for g in graded
                          if g["model"] == model and g["type"] == ttype and g["cond"] == "on"}
            off_by_item = {g["item_id"]: g for g in graded
                           if g["model"] == model and g["type"] == ttype and g["cond"] == "off"}
            shared = sorted(set(on_by_item) & set(off_by_item))
            diffs = [float(on_by_item[i]["correct"]) - float(off_by_item[i]["correct"])
                     for i in shared]
            pd = paired_diff(diffs)

            # The same paired comparison restricted to items where BOTH conditions
            # produced an answer. This is the "when it answered at all, did the
            # reasoning help" question, and it is the one that separates a model
            # reasoning itself into a wrong answer from a model running out of
            # tokens. Both numbers are published because neither alone is the story.
            both = [i for i in shared
                    if on_by_item[i]["usable"] and off_by_item[i]["usable"]]
            diffs_both = [float(on_by_item[i]["correct"]) - float(off_by_item[i]["correct"])
                          for i in both]
            pd_both = paired_diff(diffs_both)

            on_trunc = by_cond["on"]["truncated"]
            off_trunc = by_cond["off"]["truncated"]
            flips = {
                "on_only": sum(1 for d in diffs if d > 0),
                "off_only": sum(1 for d in diffs if d < 0),
                "agree": sum(1 for d in diffs if d == 0),
            }
            tokens = ratio_ci(
                [g["gen_tokens"] for g in on_by_item.values()],
                [g["gen_tokens"] for g in off_by_item.values()],
            )
            wall = ratio_ci(
                [g["wall_s"] for g in on_by_item.values()],
                [g["wall_s"] for g in off_by_item.values()],
            )
            gen = ratio_ci(
                [g["gen_s"] for g in on_by_item.values()],
                [g["gen_s"] for g in off_by_item.values()],
            )
            extra_s = gen["a_mean"] - gen["b_mean"]
            cells.append({
                "model": model,
                "type": ttype,
                "off": by_cond["off"],
                "on": by_cond["on"],
                "paired": pd,
                "paired_usable_only": pd_both,
                "flips": flips,
                "verdict": (
                    "helps" if pd["significant"] and pd["mean"] > 0 else
                    "hurts" if pd["significant"] and pd["mean"] < 0 else
                    "indistinguishable"
                ),
                # A cell where thinking on ran out of budget on at least a tenth of
                # the items, more often than thinking off did. The accuracy drop
                # there is mostly missing output rather than bad reasoning, and
                # saying "thinking hurt" without saying that would be misleading.
                "budget_limited": (on_trunc / by_cond["on"]["n"] >= 0.10
                                   and on_trunc > off_trunc),
                "token_ratio": tokens["ratio"],
                "tokens": tokens,
                "wall": wall,
                "gen": gen,
                "gen_s_ratio": gen["ratio"],
                "extra_s_per_item": extra_s,
                "extra_tokens_per_item": tokens["a_mean"] - tokens["b_mean"],
                "acc_points_per_extra_second": (
                    (pd["mean"] * 100.0 / extra_s) if extra_s > 1e-9 else None
                ),
                # The per-token version is the one to trust on a shared card.
                # Seconds move with whatever else is running; tokens do not.
                "acc_points_per_1k_extra_tokens": (
                    (pd["mean"] * 100.0 * 1000.0 / (tokens["a_mean"] - tokens["b_mean"]))
                    if (tokens["a_mean"] - tokens["b_mean"]) > 1e-9 else None
                ),
            })

    return {
        "meta": {
            "temperature": 0.0,
            "top_k": 1,
            "seed": 7,
            "num_predict": 4096,
            "n_items_per_type": 40,
            "task_types": TASK_TYPES,
            "models": models,
            "interval": "95 percent, Wilson for single accuracies, paired normal for differences",
        },
        "flag_check": evidence,
        "incomplete_cells": incomplete,
        "cells": cells,
        "totals": {
            model: {
                cond: {
                    "n": sum(1 for g in graded if g["model"] == model and g["cond"] == cond),
                    "correct": sum(1 for g in graded
                                   if g["model"] == model and g["cond"] == cond and g["correct"]),
                    "usable": sum(1 for g in graded
                                  if g["model"] == model and g["cond"] == cond and g["usable"]),
                    "gen_tokens": sum(g["gen_tokens"] for g in graded
                                      if g["model"] == model and g["cond"] == cond),
                    "wall_s": sum(g["wall_s"] for g in graded
                                  if g["model"] == model and g["cond"] == cond),
                    "gen_s": sum(g["gen_s"] for g in graded
                                 if g["model"] == model and g["cond"] == cond),
                    "truncated": sum(1 for g in graded
                                     if g["model"] == model and g["cond"] == cond
                                     and g["done_reason"] == "length"),
                }
                for cond in ("off", "on")
            }
            for model in models
        },
    }


def main() -> None:
    summary = analyze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    bad = [m for m, e in summary["flag_check"].items() if not e["flag_effective"]]
    if bad:
        raise SystemExit(
            "the think flag did not take effect for: " + ", ".join(bad) +
            " (traces present with it on and absent with it off is the requirement)"
        )
    print(f"wrote {OUT.name}: {len(summary['cells'])} model x task-type cells")
    for c in summary["cells"]:
        print(f"  {c['model']:14s} {c['type']:10s} "
              f"off {c['off']['accuracy']*100:5.1f}%  on {c['on']['accuracy']*100:5.1f}%  "
              f"diff {c['paired']['mean']*100:+6.1f} "
              f"[{c['paired']['lo']*100:+.1f},{c['paired']['hi']*100:+.1f}] "
              f"{c['verdict']}  x{c['token_ratio']:.1f} tokens")


if __name__ == "__main__":
    main()
