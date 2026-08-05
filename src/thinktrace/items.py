"""Deterministic generation of the fixed evaluation item set.

Five task types, chosen to be different in kind so that "thinking helps" is not
assumed to be uniform:

  arith      multi-step arithmetic word problems (serial computation)
  deduct     logical deduction over ordering constraints (search)
  recall     single-fact factual recall (retrieval, no computation)
  instruct   output-format instruction following (constraint satisfaction)
  overthink  altered classic puzzles whose literal reading is trivially correct
             and whose memorised look-alike answer is wrong

Every item is produced from a fixed seed, so `python3 -m thinktrace.items` must
reproduce `data/items.json` byte for byte. The verify script asserts that.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

ANSWER_HINT = "End your reply with a line of the form 'Answer: <answer>'."

N_PER_TYPE = 40
SEED = 20260805

TASK_TYPES = ["arith", "deduct", "recall", "instruct", "overthink"]


# --------------------------------------------------------------------------
# arith
# --------------------------------------------------------------------------

def _arith_items(rng: random.Random) -> list[dict]:
    # Each template carries its own parameter ranges. Shared ranges do not work:
    # `a - b*c + d` with a three digit product is negative for every draw, which
    # is an infinite rejection loop rather than a hard error.
    ranges = {
        0: ((11, 49), (7, 29), (13, 97), (4, 23)),
        1: ((11, 49), (7, 29), (13, 97), (4, 23)),
        2: ((11, 49), (7, 29), (13, 97), (4, 23)),
        3: ((400, 950), (5, 25), (3, 9), (10, 99)),
        4: ((11, 49), (17, 39), (2, 9), (2, 11)),
        5: ((41, 99), (7, 29), (13, 45), (4, 23)),
        6: ((11, 49), (7, 29), (13, 97), (4, 23)),
        7: ((3, 9), (17, 49), (2, 5), (13, 97)),
    }
    templates = [
        (
            "A warehouse holds {a} crates. Each crate contains {b} boxes. "
            "{c} boxes are removed for inspection, then {d} more full crates arrive. "
            "How many boxes are in the warehouse now?",
            lambda a, b, c, d: a * b - c + d * b,
        ),
        (
            "A printer produces {a} pages per minute for {b} minutes, then jams and "
            "wastes {c} pages, then produces {d} more pages. How many good pages were produced?",
            lambda a, b, c, d: a * b - c + d,
        ),
        (
            "A shop sells {a} shirts at {b} dollars each and {c} hats at {d} dollars each. "
            "How much money does the shop take in?",
            lambda a, b, c, d: a * b + c * d,
        ),
        (
            "A tank holds {a} litres. {b} litres are drained each hour for {c} hours, "
            "then {d} litres are added. How many litres are in the tank?",
            lambda a, b, c, d: a - b * c + d,
        ),
        (
            "A team of {a} people each collect {b} samples. {c} people then each lose {d} samples. "
            "How many samples does the team still have?",
            lambda a, b, c, d: a * b - c * d,
        ),
        (
            "A bus starts with {a} passengers. At the first stop {b} get off and {c} get on. "
            "At the second stop half of the passengers get off, then {d} get on. "
            "How many passengers are on the bus?",
            lambda a, b, c, d: (a - b + c) // 2 + d,
        ),
        (
            "A library buys {a} books per month for {b} months, donates {c} books, "
            "and then buys {d} more. How many books did it add in total?",
            lambda a, b, c, d: a * b - c + d,
        ),
        (
            "A factory runs {a} machines. Each machine makes {b} parts per shift. "
            "There are {c} shifts, and {d} parts fail quality control. "
            "How many parts pass quality control?",
            lambda a, b, c, d: a * b * c - d,
        ),
    ]
    items: list[dict] = []
    for t_index, (text, fn) in enumerate(templates):
        (ra, rb, rc, rd) = ranges[t_index]
        for k in range(N_PER_TYPE // len(templates)):
            for attempt in range(10000):
                a = rng.randint(*ra)
                b = rng.randint(*rb)
                c = rng.randint(*rc)
                d = rng.randint(*rd)
                if t_index == 5 and (a - b + c) % 2 != 0:
                    continue  # keep the halving exact so the answer is an integer
                value = fn(a, b, c, d)
                if value > 0:
                    break
            else:
                raise RuntimeError(f"no positive draw for arithmetic template {t_index}")
            items.append(
                {
                    "id": f"arith-{t_index:02d}-{k}",
                    "type": "arith",
                    "prompt": text.format(a=a, b=b, c=c, d=d) + " " + ANSWER_HINT,
                    "answer": str(value),
                    "accept": [],
                    "grader": "numeric",
                }
            )
    return items


# --------------------------------------------------------------------------
# deduct
# --------------------------------------------------------------------------

_RUNNERS = ["Ana", "Ben", "Cara", "Dan", "Eve"]


def _constraints_true_of(order: list[str]) -> list[tuple[str, str]]:
    """All candidate constraints satisfied by `order`, as (text, key) pairs."""
    pos = {name: i for i, name in enumerate(order)}
    out: list[tuple[str, str]] = []
    for x, y in itertools.permutations(_RUNNERS, 2):
        if pos[x] < pos[y]:
            out.append((f"{x} finished before {y}.", f"before:{x}:{y}"))
        if pos[x] + 1 == pos[y]:
            out.append((f"{x} finished immediately before {y}.", f"imm:{x}:{y}"))
    for name in _RUNNERS:
        for k in range(5):
            if pos[name] != k:
                out.append(
                    (f"{name} did not finish in position {k + 1}.", f"not:{name}:{k}")
                )
    return out


def _satisfies(order: list[str], key: str) -> bool:
    pos = {name: i for i, name in enumerate(order)}
    kind, *rest = key.split(":")
    if kind == "before":
        return pos[rest[0]] < pos[rest[1]]
    if kind == "imm":
        return pos[rest[0]] + 1 == pos[rest[1]]
    if kind == "not":
        return pos[rest[0]] != int(rest[1])
    raise ValueError(key)


def _deduct_items(rng: random.Random) -> list[dict]:
    all_orders = [list(p) for p in itertools.permutations(_RUNNERS)]
    items: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(items) < N_PER_TYPE:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("could not build a unique-solution deduction set")
        target = list(rng.choice(all_orders))
        pool = _constraints_true_of(target)
        rng.shuffle(pool)
        chosen: list[tuple[str, str]] = []
        survivors = all_orders
        for text, key in pool:
            if len(chosen) >= 6:
                break
            nxt = [o for o in survivors if _satisfies(o, key)]
            if len(nxt) == len(survivors):
                continue  # constraint eliminates nothing, skip it
            chosen.append((text, key))
            survivors = nxt
            if len(survivors) == 1:
                break
        if len(survivors) != 1:
            continue
        sig = tuple(sorted(k for _, k in chosen))
        if sig in seen:
            continue
        seen.add(sig)
        ask = rng.randint(0, 4)
        clue_text = " ".join(t for t, _ in chosen)
        prompt = (
            "Five runners finished a race: Ana, Ben, Cara, Dan and Eve. "
            f"{clue_text} "
            f"Who finished in position {ask + 1}? {ANSWER_HINT}"
        )
        items.append(
            {
                "id": f"deduct-{len(items):02d}",
                "type": "deduct",
                "prompt": prompt,
                "answer": target[ask],
                "accept": [],
                "grader": "text",
            }
        )
    return items


# --------------------------------------------------------------------------
# recall
# --------------------------------------------------------------------------

_RECALL: list[tuple[str, str, list[str]]] = [
    ("What is the chemical symbol for potassium?", "K", []),
    ("What is the chemical symbol for tungsten?", "W", []),
    ("What is the chemical symbol for lead?", "Pb", []),
    ("What is the chemical symbol for tin?", "Sn", []),
    ("What is the chemical symbol for silver?", "Ag", []),
    ("What is the capital city of Australia?", "Canberra", []),
    ("What is the capital city of Canada?", "Ottawa", []),
    ("What is the capital city of Brazil?", "Brasilia", ["Brasília"]),
    ("What is the capital city of Switzerland?", "Bern", ["Berne"]),
    ("What is the capital city of Turkey?", "Ankara", []),
    ("What is the capital city of Vietnam?", "Hanoi", ["Ha Noi"]),
    ("What is the capital city of Morocco?", "Rabat", []),
    ("What is the official currency of Japan?", "Yen", ["Japanese yen"]),
    ("What is the official currency of Poland?", "Zloty", ["Polish zloty", "Złoty"]),
    ("What is the official currency of South Korea?", "Won", ["South Korean won"]),
    ("What is the official currency of India?", "Rupee", ["Indian rupee"]),
    ("In which year did the Berlin Wall fall?", "1989", []),
    ("In which year did the first human walk on the Moon?", "1969", []),
    ("In which year did the Titanic sink?", "1912", []),
    ("In which year did the Second World War end?", "1945", []),
    ("Who wrote the novel 'Nineteen Eighty-Four'?", "George Orwell", ["Orwell"]),
    ("Who wrote the play 'A Doll's House'?", "Henrik Ibsen", ["Ibsen"]),
    ("Who wrote the novel 'Things Fall Apart'?", "Chinua Achebe", ["Achebe"]),
    ("Who painted 'The Starry Night'?", "Vincent van Gogh", ["van Gogh", "Van Gogh"]),
    ("Which planet is closest to the Sun?", "Mercury", []),
    ("Which planet has the most moons of any in the Solar System?", "Saturn", []),
    ("Which is the largest ocean on Earth?", "Pacific", ["Pacific Ocean"]),
    ("Which is the longest river in South America?", "Amazon", ["Amazon River"]),
    ("What is the largest country in Africa by land area?", "Algeria", []),
    ("What is the tallest mountain above sea level?", "Everest", ["Mount Everest"]),
    ("How many bones are in the adult human body?", "206", []),
    ("How many players are on the field per team in a football (soccer) match?", "11", ["eleven"]),
    ("How many strings does a standard violin have?", "4", ["four"]),
    ("How many sides does a dodecagon have?", "12", ["twelve"]),
    ("What is the hardest naturally occurring mineral?", "Diamond", []),
    ("What gas do plants absorb from the air for photosynthesis?", "Carbon dioxide", ["CO2"]),
    ("What is the SI base unit of electric current?", "Ampere", ["Amp", "A"]),
    ("What is the boiling point of water at sea level in degrees Celsius?", "100", []),
    ("Which language has the most native speakers worldwide?", "Mandarin", ["Mandarin Chinese", "Chinese"]),
    ("What is the smallest prime number?", "2", ["two"]),
]


def _recall_items() -> list[dict]:
    items = []
    for i, (q, ans, accept) in enumerate(_RECALL):
        items.append(
            {
                "id": f"recall-{i:02d}",
                "type": "recall",
                "prompt": f"{q} {ANSWER_HINT}",
                "answer": ans,
                "accept": accept,
                "grader": "text",
            }
        )
    return items


# --------------------------------------------------------------------------
# instruct
# --------------------------------------------------------------------------

def _instruct_items(rng: random.Random) -> list[dict]:
    items: list[dict] = []

    def add(prompt: str, check: str, params: dict) -> None:
        items.append(
            {
                "id": f"instruct-{len(items):02d}",
                "type": "instruct",
                "prompt": prompt,
                "answer": "",
                "accept": [],
                "grader": "constraint",
                "check": check,
                "params": params,
            }
        )

    topics = ["rain", "a bicycle", "the ocean", "a library", "winter"]
    for i, n in enumerate([5, 7, 9, 11, 13]):
        add(
            f"Write a description of {topics[i]} that is exactly {n} words long. "
            "Reply with the description only, no other text.",
            "word_count",
            {"n": n},
        )
    for i, n in enumerate([3, 4, 5, 6, 8]):
        word = ["ripple", "anchor", "lantern", "meadow", "signal"][i]
        add(
            f"Reply with the word '{word}' repeated exactly {n} times, "
            "separated by single spaces, and nothing else.",
            "repeat_word",
            {"word": word, "n": n},
        )
    for n in [3, 4, 5, 6, 7]:
        add(
            f"Name {n} colours in capital letters, separated by commas, and nothing else. "
            "Do not add a full stop.",
            "upper_csv",
            {"n": n},
        )
    for keys in [["a", "b"], ["x", "y", "z"], ["one", "two"], ["p", "q", "r"], ["k", "v"]]:
        add(
            "Reply with a single JSON object and nothing else. It must have exactly the keys "
            + ", ".join(f"\"{k}\"" for k in keys)
            + " in that order, and every value must be the integer 1.",
            "json_keys",
            {"keys": keys},
        )
    for n in [4, 6, 8, 10, 12]:
        add(
            f"Reply with exactly {n} asterisk characters on one line and nothing else.",
            "exact_chars",
            {"char": "*", "n": n},
        )
    for i, (lo, hi) in enumerate([(3, 9), (5, 12), (2, 8), (7, 15), (4, 11)]):
        add(
            f"Reply with the whole numbers from {hi} down to {lo} in descending order, "
            "separated by commas with no spaces, and nothing else.",
            "desc_range",
            {"lo": lo, "hi": hi},
        )
    for i, start in enumerate(["Blue", "Quiet", "Seven", "Morning", "Copper"]):
        end = ["today", "again", "slowly", "north", "twice"][i]
        add(
            f"Reply with a single sentence that starts with the word '{start}' and ends with "
            f"the word '{end}'. Reply with the sentence only.",
            "start_end",
            {"start": start, "end": end},
        )
    for letter, subject in [
        ("e", "a storm"),
        ("a", "the moon"),
        ("o", "a river"),
        ("i", "a garden"),
        ("s", "the desert"),
    ]:
        add(
            f"Write one sentence of at least five words about {subject} that does not contain "
            f"the letter '{letter}' anywhere. Reply with the sentence only.",
            "forbidden_letter",
            {"letter": letter, "min_words": 5},
        )
    assert len(items) == N_PER_TYPE, len(items)
    return items


# --------------------------------------------------------------------------
# overthink
# --------------------------------------------------------------------------

_WEIGHT_OPTIONS = {
    "feathers": ["feathers", "feather"],
    "lead": ["lead"],
    "same": ["same", "equal", "equally", "neither", "identical", "tie"],
}
_PLACE_OPTIONS = {
    "first": ["first", "1st"],
    "second": ["second", "2nd"],
    "third": ["third", "3rd"],
    "fourth": ["fourth", "4th"],
    "fifth": ["fifth", "5th"],
    "last": ["last"],
}


def _overthink_items(rng: random.Random) -> list[dict]:
    items: list[dict] = []

    def add(
        prompt: str,
        answer: str,
        accept: list[str],
        grader: str = "text",
        options: dict | None = None,
    ) -> None:
        item = {
            "id": f"overthink-{len(items):02d}",
            "type": "overthink",
            "prompt": f"{prompt} {ANSWER_HINT}",
            "answer": answer,
            "accept": accept,
            "grader": grader,
        }
        if options is not None:
            item["options"] = options
        items.append(item)

    # "all but N" - the literal answer is N; the memorised arithmetic answer is total - N.
    for total, left in [(17, 9), (23, 6), (31, 12), (14, 5), (28, 7), (19, 11), (36, 4), (25, 8)]:
        add(
            f"A farmer has {total} sheep. All but {left} run away. "
            "How many sheep does the farmer have left?",
            str(left),
            [],
            "numeric",
        )
    # Weight comparisons. The famous version has equal masses and answer "the same",
    # so the memorised pull is towards "the same" while the literal answer is not.
    for heavy, light in [(3, 1), (5, 2), (7, 4), (9, 6)]:
        add(
            f"Which weighs more, {heavy} kilograms of feathers or {light} kilograms of lead?",
            "feathers",
            [],
            "choice",
            _WEIGHT_OPTIONS,
        )
    for heavy, light in [(4, 2), (10, 3), (6, 5), (8, 1)]:
        add(
            f"Which weighs more, {light} kilograms of feathers or {heavy} kilograms of lead?",
            "lead",
            [],
            "choice",
            _WEIGHT_OPTIONS,
        )
    # Race position. Overtaking the runner in Nth place puts you in Nth place.
    for place in ["second", "third", "fourth", "fifth"]:
        add(
            f"You are running a race and you overtake the runner in {place} place. "
            "What place are you in now?",
            place,
            [],
            "choice",
            _PLACE_OPTIONS,
        )
    # months with N days
    add("How many months of the year have 28 days?", "12", ["twelve", "all"], "numeric")
    add("How many months of the year have at least 30 days?", "11", ["eleven"], "numeric")
    # named-sibling puzzle
    for name in ["Mia", "Priya", "Rosa", "Hana"]:
        add(
            f"{name}'s mother has four daughters. Three of them are named April, May and June. "
            "What is the fourth daughter's name?",
            name,
            [],
        )
    # literal counting with a distracting frame
    for word, letter, count in [
        ("bookkeeper", "k", 3),
        ("Mississippi", "s", 4),
        ("banana", "a", 3),
        ("committee", "t", 2),
    ]:
        add(
            f"How many times does the letter '{letter}' appear in the word '{word}'?",
            str(count),
            [],
            "numeric",
        )
    # obvious-answer questions dressed up as puzzles
    add(
        "A butcher is 175 centimetres tall and wears size 44 shoes. What does he weigh?",
        "meat",
        [],
    )
    add(
        "If a red house is made of red bricks and a blue house is made of blue bricks, "
        "what is a greenhouse made of?",
        "glass",
        [],
    )
    add(
        "A plane crashes exactly on the border between two countries. "
        "Where are the survivors buried?",
        "nowhere",
        [
            "not buried",
            "do not bury",
            "don't bury",
            "you do not bury the survivors",
            "survivors are not buried",
            "they are alive",
        ],
    )
    add(
        "Some months have 31 days and some have 30. How many have 31 days?",
        "7",
        ["seven"],
        "numeric",
    )
    add(
        "A rooster lays an egg on the exact peak of a roof. Which side does the egg roll down?",
        "neither",
        [],
        "choice",
        {
            "left": ["left"],
            "right": ["right"],
            "neither": [
                "neither",
                "no side",
                "none",
                "roosters do not lay",
                "roosters don't lay",
                "does not lay",
                "doesn't lay",
                "cannot lay",
                "no egg",
            ],
        },
    )
    add(
        "Divide 30 by one half and add 10. What is the result?",
        "70",
        [],
        "numeric",
    )
    add(
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would 100 machines "
        "take to make 100 widgets?",
        "5",
        ["five", "5 minutes"],
        "numeric",
    )
    add(
        "Before Mount Everest was discovered, what was the tallest mountain on Earth?",
        "Everest",
        ["Mount Everest", "Everest, it was still the tallest"],
    )
    add(
        "You enter a dark room carrying one match. The room contains an oil lamp, "
        "a candle and a wood fire. What do you light first?",
        "match",
        [],
        "choice",
        {
            "match": ["match", "matchstick"],
            "lamp": ["lamp", "oil lamp"],
            "candle": ["candle"],
            "fire": ["fire", "wood fire", "fireplace"],
        },
    )
    add(
        "How many animals of each kind did Moses take onto the ark?",
        "none",
        ["zero", "0", "moses did not", "it was noah", "noah not moses"],
    )
    assert len(items) == N_PER_TYPE, len(items)
    return items


# --------------------------------------------------------------------------

def build_items() -> list[dict]:
    rng = random.Random(SEED)
    items = (
        _arith_items(rng)
        + _deduct_items(rng)
        + _recall_items()
        + _instruct_items(rng)
        + _overthink_items(rng)
    )
    for t in TASK_TYPES:
        n = sum(1 for it in items if it["type"] == t)
        assert n == N_PER_TYPE, f"{t}: {n}"
    ids = [it["id"] for it in items]
    assert len(set(ids)) == len(ids), "duplicate item ids"
    return items


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "data" / "items.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    items = build_items()
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items to {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
