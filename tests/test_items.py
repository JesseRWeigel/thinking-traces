"""Item set tests.

The deduction ground truth is checked by parsing the constraints back out of the
English prompt and brute forcing all 120 orderings. That path shares no code with
the generator, so it catches a generator that emits a puzzle with two solutions or
labels the wrong runner.
"""

import itertools
import json
import re
import unittest
from pathlib import Path

from thinktrace.items import N_PER_TYPE, TASK_TYPES, build_items

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Ana", "Ben", "Cara", "Dan", "Eve"]

BEFORE = re.compile(r"(\w+) finished before (\w+)\.")
IMMEDIATE = re.compile(r"(\w+) finished immediately before (\w+)\.")
NOT_POS = re.compile(r"(\w+) did not finish in position (\d)\.")
ASK = re.compile(r"Who finished in position (\d)\?")


def solve_deduction(prompt: str) -> list[str]:
    """Brute force every ordering consistent with the prompt's clues."""
    befores = BEFORE.findall(prompt)
    immediates = IMMEDIATE.findall(prompt)
    # "X finished immediately before Y." also matches BEFORE loosely, so remove
    # the immediate pairs from the plain-before list only when the sentence was
    # literally the immediate one.
    immediate_sentences = set(IMMEDIATE.findall(prompt))
    befores = [p for p in befores if p not in immediate_sentences or
               prompt.count(f"{p[0]} finished before {p[1]}.") > 0]
    nots = NOT_POS.findall(prompt)
    out = []
    for perm in itertools.permutations(NAMES):
        pos = {n: i for i, n in enumerate(perm)}
        ok = all(pos[a] < pos[b] for a, b in befores)
        ok = ok and all(pos[a] + 1 == pos[b] for a, b in immediates)
        ok = ok and all(pos[a] != int(k) - 1 for a, k in nots)
        if ok:
            out.append(list(perm))
    return out


class TestItemSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = build_items()

    def test_shape(self):
        self.assertEqual(len(self.items), N_PER_TYPE * len(TASK_TYPES))
        for t in TASK_TYPES:
            self.assertEqual(sum(1 for i in self.items if i["type"] == t), N_PER_TYPE, t)

    def test_ids_are_unique(self):
        ids = [i["id"] for i in self.items]
        self.assertEqual(len(set(ids)), len(ids))

    def test_prompts_are_unique(self):
        prompts = [i["prompt"] for i in self.items]
        self.assertEqual(len(set(prompts)), len(prompts))

    def test_generation_is_deterministic(self):
        self.assertEqual(build_items(), build_items())

    def test_committed_file_matches_the_generator(self):
        on_disk = json.loads((ROOT / "data" / "items.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, self.items,
                         "data/items.json is stale, regenerate with python3 -m thinktrace.items")

    def test_every_deduction_puzzle_has_exactly_one_solution(self):
        for item in [i for i in self.items if i["type"] == "deduct"]:
            sols = solve_deduction(item["prompt"])
            self.assertEqual(len(sols), 1, f"{item['id']} has {len(sols)} solutions")

    def test_every_deduction_answer_is_the_solution(self):
        for item in [i for i in self.items if i["type"] == "deduct"]:
            sols = solve_deduction(item["prompt"])
            ask = int(ASK.search(item["prompt"]).group(1)) - 1
            self.assertEqual(sols[0][ask], item["answer"], item["id"])

    def test_negative_control_the_solver_rejects_a_wrong_label(self):
        # If the solver agreed with anything, the two tests above would be vacuous.
        item = [i for i in self.items if i["type"] == "deduct"][0]
        sols = solve_deduction(item["prompt"])
        ask = int(ASK.search(item["prompt"]).group(1)) - 1
        wrong = next(n for n in NAMES if n != sols[0][ask])
        self.assertNotEqual(sols[0][ask], wrong)

    def test_negative_control_an_unconstrained_puzzle_has_many_solutions(self):
        self.assertEqual(len(solve_deduction("Who finished in position 1?")), 120)

    def test_letter_count_ground_truth_is_recomputable_from_the_prompt(self):
        # A hand typed count was wrong once here (bookkeeper labelled with three k
        # when it has two), so the count is re-derived from the words in the prompt.
        rx = re.compile(r"the letter '(\w)' appear in the word '(\w+)'")
        found = 0
        for item in [i for i in self.items if i["type"] == "overthink"]:
            m = rx.search(item["prompt"])
            if not m:
                continue
            found += 1
            letter, word = m.group(1), m.group(2)
            self.assertEqual(int(item["answer"]), word.lower().count(letter.lower()),
                             f"{item['id']} {word}/{letter}")
        self.assertGreaterEqual(found, 4, "letter counting items disappeared")

    def test_negative_control_a_wrong_letter_count_would_be_caught(self):
        self.assertNotEqual("bookkeeper".count("k"), 3)
        self.assertEqual("bookkeeper".count("k"), 2)

    def test_arithmetic_answers_are_positive_integers(self):
        for item in [i for i in self.items if i["type"] == "arith"]:
            self.assertRegex(item["answer"], r"^\d+$", item["id"])
            self.assertGreater(int(item["answer"]), 0)

    def test_every_item_carries_a_grader_the_code_knows(self):
        from thinktrace.grade import GRADERS
        for item in self.items:
            self.assertIn(item["grader"], GRADERS, item["id"])
            if item["grader"] == "choice":
                self.assertIn(item["answer"], item["options"], item["id"])

    def test_choice_options_are_mutually_exclusive_on_their_own_labels(self):
        from thinktrace.grade import grade_choice
        for item in [i for i in self.items if i["grader"] == "choice"]:
            for key in item["options"]:
                probe = dict(item)
                probe["answer"] = key
                self.assertTrue(grade_choice(probe, item["options"][key][0]),
                                f"{item['id']} option {key} does not select itself")

    def test_answer_format_hint_is_on_every_graded_answer_item(self):
        for item in self.items:
            if item["grader"] == "constraint":
                self.assertNotIn("Answer:", item["prompt"], item["id"])
            else:
                self.assertIn("Answer: <answer>", item["prompt"], item["id"])


if __name__ == "__main__":
    unittest.main()
