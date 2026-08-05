#!/usr/bin/env python3
"""Stamp `prompt_sha` onto cached responses collected before that field existed.

The fingerprint is a factual claim: this response was produced from this exact
prompt. So this tool refuses to guess. It only stamps a record when the prompt in
the current item set is byte identical to the prompt in the git revision named on
the command line, which is the revision the responses were collected against.

Any item whose prompt changed since then is left unstamped and named on stdout.
Those responses are stale and have to be deleted and re-run. `verify.sh` fails on
a response with no fingerprint, so an unstamped record cannot quietly ship.

    python3 tools/backfill_prompt_sha.py <git-rev-items-were-collected-against>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thinktrace.items import build_items  # noqa: E402
from thinktrace.runner import prompt_sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    rev = sys.argv[1]
    proc = subprocess.run(
        ["git", "show", f"{rev}:data/items.json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        print(f"cannot read data/items.json at {rev}: {proc.stderr.strip()}")
        return 2
    old = {i["id"]: i["prompt"] for i in json.loads(proc.stdout)}
    new = {i["id"]: i["prompt"] for i in build_items()}

    unchanged = {k for k in new if k in old and old[k] == new[k]}
    changed = sorted(k for k in new if k in old and old[k] != new[k])
    print(f"prompts unchanged since {rev}: {len(unchanged)}")
    if changed:
        print(f"prompts CHANGED since {rev}, responses for these are stale: {changed}")

    stamped = skipped = 0
    for path in sorted((ROOT / "data" / "raw").glob("*.jsonl")):
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "prompt_sha" not in rec:
                if rec["item_id"] in unchanged:
                    ordered = {"item_id": rec["item_id"],
                               "prompt_sha": prompt_sha(new[rec["item_id"]])}
                    ordered.update({k: v for k, v in rec.items() if k != "item_id"})
                    rec = ordered
                    stamped += 1
                else:
                    skipped += 1
            out.append(json.dumps(rec, ensure_ascii=False))
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"stamped {stamped} records, left {skipped} unstamped (delete and re-run those)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
