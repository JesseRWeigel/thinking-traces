"""End to end tests for the aggregation, driven by synthetic cached responses.

These build a raw directory whose correct answer count is known by construction,
then assert the aggregate reproduces it. Each assertion is paired with a control
that perturbs the input and requires the aggregate to move.
"""

import json
import tempfile
import unittest
from pathlib import Path

from thinktrace import analyze as A
from thinktrace.items import build_items


def write_raw(dirpath: Path, records: list[dict]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list[dict]] = {}
    for r in records:
        key = f"{r['model'].replace(':', '-')}__{'think-on' if r['think_requested'] else 'think-off'}"
        by_file.setdefault(key, []).append(r)
    for key, rows in by_file.items():
        (dirpath / f"{key}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )


def record(item, think, content, thinking="", tokens=10, done="stop", gen_s=1.0):
    return {
        "item_id": item["id"],
        "type": item["type"],
        "model": "fake:1b",
        "think_requested": think,
        "content": content,
        "thinking": thinking,
        "done_reason": done,
        "eval_count": tokens,
        "prompt_eval_count": 20,
        "total_duration_ns": int(gen_s * 1e9),
        "eval_duration_ns": int(gen_s * 1e9),
        "load_duration_ns": 0,
        "wall_s": gen_s,
        "num_predict": 4096,
        "temperature": 0.0,
    }


class AnalyzeCase(unittest.TestCase):
    def setUp(self):
        self.items = build_items()
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name) / "raw"
        self.rep = Path(self.tmp.name) / "replicate"
        self._orig = A.RAW_DIR
        self._orig_rep = A.REPLICATE_DIR
        A.RAW_DIR = self.raw
        A.REPLICATE_DIR = self.rep

    def tearDown(self):
        A.RAW_DIR = self._orig
        A.REPLICATE_DIR = self._orig_rep
        self.tmp.cleanup()

    def arith(self, n):
        return [i for i in self.items if i["type"] == "arith"][:n]


class TestAccuracy(AnalyzeCase):
    def build(self, n_right_off, n_right_on, n=10):
        items = self.arith(n)
        recs = []
        for i, item in enumerate(items):
            off_ok = i < n_right_off
            on_ok = i < n_right_on
            recs.append(record(item, False,
                               f"Answer: {item['answer'] if off_ok else 0}"))
            recs.append(record(item, True,
                               f"Answer: {item['answer'] if on_ok else 0}",
                               thinking="let me think about it", tokens=100))
        write_raw(self.raw, recs)
        return A.analyze()

    def test_accuracy_matches_construction(self):
        s = self.build(3, 7)
        cell = next(c for c in s["cells"] if c["type"] == "arith")
        self.assertEqual(cell["off"]["correct"], 3)
        self.assertEqual(cell["on"]["correct"], 7)
        self.assertAlmostEqual(cell["off"]["accuracy"], 0.3)
        self.assertAlmostEqual(cell["on"]["accuracy"], 0.7)

    def test_negative_control_a_different_construction_gives_different_numbers(self):
        cell_a = next(c for c in self.build(3, 7)["cells"] if c["type"] == "arith")
        self.tearDown()
        self.setUp()
        cell_b = next(c for c in self.build(8, 2)["cells"] if c["type"] == "arith")
        self.assertNotEqual(cell_a["off"]["correct"], cell_b["off"]["correct"])
        self.assertNotEqual(cell_a["verdict"], cell_b["verdict"])

    def test_paired_difference_matches_the_flips(self):
        s = self.build(3, 7)
        cell = next(c for c in s["cells"] if c["type"] == "arith")
        self.assertEqual(cell["flips"]["on_only"], 4)
        self.assertEqual(cell["flips"]["off_only"], 0)
        self.assertAlmostEqual(cell["paired"]["mean"], 0.4)

    def test_a_regression_is_reported_as_hurting(self):
        s = self.build(10, 0)
        cell = next(c for c in s["cells"] if c["type"] == "arith")
        self.assertEqual(cell["verdict"], "hurts")
        self.assertLess(cell["paired"]["mean"], 0)

    def test_no_difference_is_reported_as_indistinguishable(self):
        s = self.build(5, 5)
        cell = next(c for c in s["cells"] if c["type"] == "arith")
        self.assertEqual(cell["verdict"], "indistinguishable")

    def test_token_ratio_is_the_ratio_of_means(self):
        s = self.build(5, 5)
        cell = next(c for c in s["cells"] if c["type"] == "arith")
        self.assertAlmostEqual(cell["token_ratio"], 10.0)


class TestUsability(AnalyzeCase):
    def test_empty_outputs_are_counted_apart_from_wrong_ones(self):
        items = self.arith(10)
        recs = []
        for i, item in enumerate(items):
            recs.append(record(item, False, f"Answer: {item['answer']}"))
            # thinking on: half produce nothing at all, half are wrong
            if i < 5:
                recs.append(record(item, True, "", thinking="x" * 500,
                                   tokens=4096, done="length"))
            else:
                recs.append(record(item, True, "Answer: 0", thinking="x" * 50, tokens=80))
        write_raw(self.raw, recs)
        cell = next(c for c in A.analyze()["cells"] if c["type"] == "arith")
        self.assertEqual(cell["on"]["usable"], 5)
        self.assertEqual(cell["on"]["truncated"], 5)
        self.assertEqual(cell["on"]["correct"], 0)
        self.assertEqual(cell["on"]["accuracy"], 0.0)

    def test_accuracy_among_usable_separates_the_two_failure_modes(self):
        # Ten items. Thinking off answers all ten correctly. Thinking on answers
        # five and gets four of them right, and produces nothing for the other five.
        # Headline accuracy is 40 percent, accuracy among usable is 80 percent, and
        # the difference between those two numbers is the whole point.
        items = self.arith(10)
        recs = []
        for i, item in enumerate(items):
            recs.append(record(item, False, f"Answer: {item['answer']}"))
            if i < 5:
                recs.append(record(item, True, "", thinking="x" * 900,
                                   tokens=4096, done="length"))
            elif i < 9:
                recs.append(record(item, True, f"Answer: {item['answer']}",
                                   thinking="x" * 90, tokens=200))
            else:
                recs.append(record(item, True, "Answer: 0", thinking="x" * 90, tokens=200))
        write_raw(self.raw, recs)
        cell = next(c for c in A.analyze()["cells"] if c["type"] == "arith")
        self.assertAlmostEqual(cell["on"]["accuracy"], 0.4)
        self.assertAlmostEqual(cell["on"]["accuracy_usable"], 0.8)
        self.assertTrue(cell["budget_limited"])
        # Over the five items answered in both conditions the difference is one loss.
        self.assertEqual(cell["paired_usable_only"]["n"], 5)
        self.assertAlmostEqual(cell["paired_usable_only"]["mean"], -0.2)

    def test_negative_control_a_cell_with_no_truncation_is_not_budget_limited(self):
        items = self.arith(10)
        recs = []
        for item in items:
            recs.append(record(item, False, f"Answer: {item['answer']}"))
            recs.append(record(item, True, "Answer: 0", thinking="x" * 90))
        write_raw(self.raw, recs)
        cell = next(c for c in A.analyze()["cells"] if c["type"] == "arith")
        self.assertFalse(cell["budget_limited"])
        self.assertEqual(cell["paired_usable_only"]["n"], 10)
        self.assertAlmostEqual(cell["on"]["accuracy_usable"], cell["on"]["accuracy"])

    def test_negative_control_usable_and_correct_are_not_the_same_number(self):
        items = self.arith(10)
        recs = []
        for item in items:
            recs.append(record(item, False, f"Answer: {item['answer']}"))
            recs.append(record(item, True, "Answer: 0", thinking="x" * 50))
        write_raw(self.raw, recs)
        cell = next(c for c in A.analyze()["cells"] if c["type"] == "arith")
        self.assertEqual(cell["on"]["usable"], 10)
        self.assertEqual(cell["on"]["correct"], 0)


class TestIncompleteRuns(AnalyzeCase):
    def test_a_condition_with_no_data_is_recorded_not_dropped(self):
        recs = [record(item, False, "Answer: 1") for item in self.arith(10)]
        write_raw(self.raw, recs)
        s = A.analyze()
        self.assertEqual(s["cells"], [])
        self.assertTrue(any(c["type"] == "arith" for c in s["incomplete_cells"]))
        self.assertEqual(s["incomplete_cells"][0]["conditions_present"], ["off"])

    def test_negative_control_a_complete_cell_is_not_recorded_as_incomplete(self):
        recs = []
        for item in self.arith(10):
            recs.append(record(item, False, "Answer: 1"))
            recs.append(record(item, True, "Answer: 1", thinking="r"))
        write_raw(self.raw, recs)
        s = A.analyze()
        self.assertTrue(any(c["type"] == "arith" for c in s["cells"]))
        self.assertNotIn("arith", [c["type"] for c in s["incomplete_cells"]])


class TestFlagEvidence(AnalyzeCase):
    def _records(self, on_thinking: str, off_thinking: str = ""):
        recs = []
        for item in self.arith(10):
            recs.append(record(item, False, "Answer: 1", thinking=off_thinking))
            recs.append(record(item, True, "Answer: 1", thinking=on_thinking))
        return recs

    def test_a_working_flag_is_recognised(self):
        write_raw(self.raw, self._records("reasoning here"))
        s = A.analyze()
        self.assertTrue(s["flag_check"]["fake:1b"]["flag_effective"])

    def test_negative_control_a_silently_ignored_flag_is_caught(self):
        # Both conditions carry no reasoning payload, which is exactly what a
        # dropped `think` field looks like. The run must be rejected.
        write_raw(self.raw, self._records(""))
        s = A.analyze()
        self.assertFalse(s["flag_check"]["fake:1b"]["flag_effective"])

    def test_negative_control_traces_in_both_conditions_is_also_caught(self):
        # A model that always reasons was never switched off.
        write_raw(self.raw, self._records("reasoning", "reasoning"))
        s = A.analyze()
        self.assertFalse(s["flag_check"]["fake:1b"]["flag_effective"])

    def test_main_refuses_to_publish_when_the_flag_never_took_effect(self):
        write_raw(self.raw, self._records(""))
        out = Path(self.tmp.name) / "summary.json"
        orig_out = A.OUT
        A.OUT = out
        try:
            with self.assertRaises(SystemExit):
                A.main()
        finally:
            A.OUT = orig_out


class TestNoiseFloor(AnalyzeCase):
    def _main_run(self, n=20):
        recs = []
        for item in self.arith(n):
            recs.append(record(item, False, f"Answer: {item['answer']}"))
            recs.append(record(item, True, f"Answer: {item['answer']}", thinking="r"))
        write_raw(self.raw, recs)
        return self.arith(n)

    def test_unmeasured_is_reported_as_unmeasured_not_as_zero(self):
        self._main_run()
        s = A.analyze()
        self.assertFalse(s["noise_floor"]["measured"])
        self.assertIn("reason", s["noise_floor"])
        # None, not False. "Could not check" and "checked, found nothing" are
        # different values and downstream code has to be able to tell them apart.
        self.assertTrue(all(c["clears_noise_floor"] is None for c in s["cells"]))

    def test_a_perfectly_repeatable_backend_has_a_floor_above_zero_anyway(self):
        # Zero observed flips still leaves an interval, because 0 of 20 does not
        # prove a zero rate. The floor is the upper bound, so it must exceed zero.
        items = self._main_run()
        reps = []
        for item in items[:10]:
            reps.append(record(item, False, f"Answer: {item['answer']}"))
            reps.append(record(item, True, f"Answer: {item['answer']}", thinking="r"))
        write_raw(self.rep, reps)
        s = A.analyze()
        floor = s["noise_floor"]
        self.assertTrue(floor["measured"])
        for cell in floor["by_condition"].values():
            self.assertEqual(cell["grade_flips"], 0)
        self.assertGreater(floor["floor_upper_bound"], 0.0)

    def test_flips_are_counted_and_raise_the_floor(self):
        items = self._main_run()
        reps = []
        for i, item in enumerate(items[:10]):
            # Half the replicated think-on responses come back wrong this time.
            good = f"Answer: {item['answer']}"
            reps.append(record(item, False, good))
            reps.append(record(item, True, good if i >= 5 else "Answer: 0", thinking="r"))
        write_raw(self.rep, reps)
        floor = A.analyze()["noise_floor"]
        on = floor["by_condition"]["fake:1b|on"]
        off = floor["by_condition"]["fake:1b|off"]
        self.assertEqual(on["grade_flips"], 5)
        self.assertEqual(off["grade_flips"], 0)
        self.assertAlmostEqual(on["grade_flip_rate"], 0.5)
        self.assertGreater(floor["floor_upper_bound"], 0.5)

    def test_negative_control_a_bigger_flip_rate_gives_a_higher_floor(self):
        items = self._main_run()
        write_raw(self.rep, [record(i, True, f"Answer: {i['answer']}", thinking="r")
                             for i in items[:10]])
        low = A.analyze()["noise_floor"]["floor_upper_bound"]
        self.tearDown()
        self.setUp()
        items = self._main_run()
        write_raw(self.rep, [record(i, True, "Answer: 0", thinking="r") for i in items[:10]])
        high = A.analyze()["noise_floor"]["floor_upper_bound"]
        self.assertGreater(high, low)

    def test_an_effect_below_the_floor_does_not_clear_it(self):
        # Main run: thinking on wins one extra item out of twenty, a 5 point effect.
        items = self.arith(20)
        recs = []
        for i, item in enumerate(items):
            recs.append(record(item, False, "Answer: 0" if i == 0 else f"Answer: {item['answer']}"))
            recs.append(record(item, True, f"Answer: {item['answer']}", thinking="r"))
        write_raw(self.raw, recs)
        # Replicate: the backend flips half its own grades, a floor far above 5 points.
        reps = []
        for i, item in enumerate(items[:10]):
            reps.append(record(item, False, f"Answer: {item['answer']}" if i >= 5 else "Answer: 0"))
            reps.append(record(item, True, f"Answer: {item['answer']}", thinking="r"))
        write_raw(self.rep, reps)
        s = A.analyze()
        cell = next(c for c in s["cells"] if c["type"] == "arith")
        self.assertAlmostEqual(cell["paired"]["mean"], 0.05)
        self.assertGreater(s["noise_floor"]["floor_upper_bound"], 0.05)
        self.assertFalse(cell["clears_noise_floor"])

    def test_negative_control_a_large_effect_clears_a_small_floor(self):
        items = self.arith(20)
        recs = []
        for item in items:
            recs.append(record(item, False, "Answer: 0"))
            recs.append(record(item, True, f"Answer: {item['answer']}", thinking="r"))
        write_raw(self.raw, recs)
        reps = []
        for item in items:
            reps.append(record(item, False, "Answer: 0"))
            reps.append(record(item, True, f"Answer: {item['answer']}", thinking="r"))
        write_raw(self.rep, reps)
        s = A.analyze()
        cell = next(c for c in s["cells"] if c["type"] == "arith")
        self.assertAlmostEqual(cell["paired"]["mean"], 1.0)
        self.assertTrue(cell["clears_noise_floor"])

    def test_replicates_for_an_unseen_condition_are_ignored_rather_than_crashing(self):
        self._main_run()
        stray = record(self.arith(1)[0], False, "Answer: 1")
        stray["model"] = "other:1b"
        write_raw(self.rep, [stray])
        floor = A.analyze()["noise_floor"]
        self.assertNotIn("other:1b|off", floor["by_condition"])


if __name__ == "__main__":
    unittest.main()
