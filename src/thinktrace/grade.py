"""Answer extraction and deterministic grading.

No model judges another model here. Every grader is a closed-form predicate over
the item's ground truth, so the same raw response always produces the same grade.

Three outcomes exist per response and they are kept apart on purpose:

  usable=False   the model produced nothing to grade (empty content, or the
                 generation hit the token cap before emitting an answer)
  usable=True,  correct=False
  usable=True,  correct=True

Accuracy is reported over all attempted items, and the usable count is reported
next to it, so "0 percent correct" can never be confused with "produced nothing".
"""

from __future__ import annotations

import json
import re
import unicodedata

FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
# Models wrap the label in markdown bold in several different places, so the
# asterisks are tolerated on either side of the colon and stripped from the value.
ANSWER_RE = re.compile(
    r"\*{0,2}\s*\banswer\s*\*{0,2}\s*[:\-]\s*\*{0,2}\s*(.*?)[\s*]*$", re.IGNORECASE
)
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}


def strip_fences(text: str) -> str:
    m = FENCE_RE.match(text or "")
    return m.group(1) if m else (text or "")


def extract_answer(content: str) -> str:
    """Pull the answer out of a response body.

    Prefers the last 'Answer: ...' line, since every prompt asks for one. Falls
    back to the last non-empty line so a model that ignores the format is still
    graded on what it said rather than scored zero for formatting.
    """
    body = strip_fences(content or "")
    lines = [ln.rstrip() for ln in body.splitlines()]
    # The label is searched for anywhere in the line, not only at its start.
    # Models routinely emit the whole response on one line, as in
    # "The currency of South Korea is the won. Answer: Won", and anchoring to the
    # line start scored several of those as unanswered.
    for line in reversed(lines):
        m = ANSWER_RE.search(line)
        if m and m.group(1).strip():
            return m.group(1).strip()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


def parse_number(s: str) -> float | None:
    norm = normalize_text(s)
    if norm in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[norm])
    m = NUM_RE.search(s or "")
    if not m:
        first = norm.split(" ")[0] if norm else ""
        if first in _WORD_NUMBERS:
            return float(_WORD_NUMBERS[first])
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _contains_phrase(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


# --------------------------------------------------------------------------
# graders
# --------------------------------------------------------------------------

def grade_numeric(item: dict, pred: str) -> bool:
    norm = normalize_text(pred)
    for cand in item.get("accept", []):
        if norm == normalize_text(cand):
            return True
    got = parse_number(pred)
    want = parse_number(item["answer"])
    if got is None or want is None:
        return False
    return abs(got - want) < 1e-9


def grade_text(item: dict, pred: str) -> bool:
    got = normalize_text(pred)
    if not got:
        return False
    candidates = [item["answer"]] + list(item.get("accept", []))
    for cand in candidates:
        want = normalize_text(cand)
        if not want:
            continue
        if got == want:
            return True
        # A short reply that spells out the answer counts. The cap keeps an essay
        # that happens to mention the right word from scoring as a correct answer.
        if len(got.split()) <= 12 and _contains_phrase(got, want):
            return True
    return False


def grade_choice(item: dict, pred: str) -> bool:
    got = normalize_text(pred)
    options: dict[str, list[str]] = item["options"]
    hits = set()
    for key, synonyms in options.items():
        for syn in synonyms:
            if _contains_phrase(got, normalize_text(syn)):
                hits.add(key)
                break
    if len(hits) != 1:
        return False  # ambiguous or absent counts as wrong
    return hits.pop() == item["answer"]


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"\s+", (s or "").strip()) if re.search(r"[A-Za-z0-9]", t)]


def grade_constraint(item: dict, pred: str) -> bool:
    check = item["check"]
    p = item["params"]
    text = strip_fences(pred or "").strip()
    if not text:
        return False
    if check == "word_count":
        return len(_tokens(text)) == p["n"]
    if check == "repeat_word":
        want = " ".join([p["word"]] * p["n"])
        return text.strip().rstrip(".").lower() == want.lower()
    if check == "upper_csv":
        parts = [x.strip() for x in text.rstrip(".").split(",")]
        return len(parts) == p["n"] and all(
            x.isalpha() and x.isupper() for x in parts
        )
    if check == "json_keys":
        try:
            obj = json.loads(text, object_pairs_hook=lambda pairs: pairs)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(obj, list):
            return False
        keys = [k for k, _ in obj]
        vals = [v for _, v in obj]
        return keys == p["keys"] and all(v == 1 for v in vals)
    if check == "exact_chars":
        return text == p["char"] * p["n"]
    if check == "desc_range":
        want = ",".join(str(n) for n in range(p["hi"], p["lo"] - 1, -1))
        return text.rstrip(".").replace(" ", "") == want
    if check == "start_end":
        toks = _tokens(text)
        if len(toks) < 2:
            return False
        first = re.sub(r"[^A-Za-z0-9']", "", toks[0])
        last = re.sub(r"[^A-Za-z0-9']", "", toks[-1])
        return first.lower() == p["start"].lower() and last.lower() == p["end"].lower()
    if check == "forbidden_letter":
        toks = _tokens(text)
        return len(toks) >= p["min_words"] and p["letter"].lower() not in text.lower()
    raise ValueError(f"unknown check: {check}")


GRADERS = {
    "numeric": grade_numeric,
    "text": grade_text,
    "choice": grade_choice,
    "constraint": grade_constraint,
}


def grade_response(item: dict, content: str, done_reason: str) -> dict:
    """Grade one raw response. Returns usable, correct and the extracted answer."""
    body = (content or "").strip()
    if item["grader"] == "constraint":
        pred = strip_fences(body).strip()
    else:
        pred = extract_answer(body)
    usable = bool(body) and bool(pred)
    if not usable:
        return {"usable": False, "correct": False, "pred": pred, "reason": done_reason}
    correct = GRADERS[item["grader"]](item, pred)
    return {"usable": True, "correct": bool(correct), "pred": pred, "reason": done_reason}
