"""The README's claims, as assertions against the committed summary.

Every sentence in the Findings section is a claim about data. Left as prose it
goes stale the moment someone re-runs the measurement, and the plausible version
creeps back in. So each claim is asserted here instead. If a future run produces
different numbers these tests fail, which is the correct outcome: it forces the
prose to be rewritten rather than quietly contradicted by the data next to it.

Each assertion is paired with a negative control showing the same check rejects
the opposite shape of data, so none of them can be passing by accident of always
being true.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestFindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
        cls.cells = cls.summary["cells"]

    def cell(self, model, ttype):
        return next(c for c in self.cells if c["model"] == model and c["type"] == ttype)

    # ---------------------------------------------------------- the headline

    def test_no_cell_shows_thinking_helping(self):
        helped = [f"{c['model']}/{c['type']}" for c in self.cells if c["verdict"] == "helps"]
        self.assertEqual(helped, [], "README says no cell helps")

    def test_no_cell_differs_over_items_answered_in_both_conditions(self):
        sig = [f"{c['model']}/{c['type']}" for c in self.cells
               if c["paired_usable_only"]["significant"]]
        self.assertEqual(sig, [], "README says the answered-only comparison is null everywhere")

    def test_negative_control_the_significance_check_can_fire(self):
        # If `significant` were always False the test above would be vacuous.
        from thinktrace.stats import paired_diff
        self.assertTrue(paired_diff([1.0] * 40)["significant"])
        self.assertFalse(paired_diff([0.0] * 40)["significant"])

    def test_every_hurting_cell_is_budget_limited(self):
        for c in self.cells:
            if c["verdict"] == "hurts":
                self.assertTrue(
                    c["budget_limited"],
                    f"{c['model']}/{c['type']} hurts without being budget limited, "
                    "which would contradict the README's explanation",
                )

    def test_negative_control_not_every_cell_is_budget_limited(self):
        # The claim above is only meaningful if the flag distinguishes cells.
        flags = {c["budget_limited"] for c in self.cells}
        self.assertEqual(flags, {True, False})

    # ------------------------------------------------------------ the costs

    def test_thinking_costs_at_least_three_times_the_tokens_everywhere(self):
        worst = min(c["token_ratio"] for c in self.cells)
        self.assertGreater(worst, 3.0, "README quotes a 3.3x floor on token cost")

    def test_instruction_following_on_the_newer_model_is_the_extreme_cost_case(self):
        c = self.cell("qwen3.5:9b", "instruct")
        self.assertGreater(c["token_ratio"], 100.0)
        self.assertEqual(max(self.cells, key=lambda x: x["token_ratio"])["type"], "instruct")

    def test_negative_control_token_ratios_are_not_all_equal(self):
        ratios = {round(c["token_ratio"], 3) for c in self.cells}
        self.assertGreater(len(ratios), 5)

    # ------------------------------------- think off still reasons, in content

    def test_think_off_still_spends_hundreds_of_tokens_on_deduction(self):
        # The README's claim that `think: false` removes the channel and not the
        # reasoning rests on these two numbers.
        self.assertGreater(self.cell("qwen3:8b", "deduct")["off"]["gen_tokens_mean"], 500)
        self.assertGreater(self.cell("qwen3.5:9b", "deduct")["off"]["gen_tokens_mean"], 900)

    def test_negative_control_think_off_is_cheap_where_there_is_nothing_to_work_out(self):
        # Same flag, same model, two orders of magnitude apart. Without this the
        # test above would read as "think-off is always expensive", which is false.
        self.assertLess(self.cell("qwen3.5:9b", "instruct")["off"]["gen_tokens_mean"], 50)

    # ---------------------------------------------------- truncation asymmetry

    def test_truncation_is_far_worse_with_thinking_on(self):
        for model, totals in self.summary["totals"].items():
            self.assertGreater(totals["on"]["truncated"], totals["off"]["truncated"], model)
        t = self.summary["totals"]["qwen3.5:9b"]
        self.assertGreaterEqual(t["on"]["truncated"], 50)
        self.assertEqual(t["off"]["truncated"], 0)

    def test_the_worst_cell_is_missing_output_rather_than_wrong_answers(self):
        c = self.cell("qwen3.5:9b", "deduct")
        self.assertLess(c["on"]["accuracy"], 0.2)
        self.assertGreater(c["on"]["accuracy_usable"], 0.8)
        self.assertLessEqual(c["on"]["usable"], 10)

    def test_the_weak_row_is_named_as_weak(self):
        # README calls out that one answered-only row is uninformative rather than
        # reassuring. If its sample ever grows, the caveat has to be rewritten.
        weak = self.cell("qwen3.5:9b", "deduct")["paired_usable_only"]
        self.assertLessEqual(weak["n"], 10)
        self.assertGreater(weak["hi"] - weak["lo"], 0.4, "README calls this interval very wide")

    def test_negative_control_the_other_rows_are_not_that_weak(self):
        others = [c for c in self.cells
                  if not (c["model"] == "qwen3.5:9b" and c["type"] == "deduct")]
        for c in others:
            pu = c["paired_usable_only"]
            self.assertGreaterEqual(pu["n"], 27, f"{c['model']}/{c['type']}")
            self.assertLess(pu["hi"] - pu["lo"], 0.4, f"{c['model']}/{c['type']}")

    # ------------------------------------------- the refuted overthink hypothesis

    def test_reasoning_does_not_talk_either_model_out_of_the_literal_answer(self):
        # The hypothesis the overthink set was built to expose. It failed, and the
        # failure is asserted here so the plausible version cannot creep back into
        # the README later.
        for model in ("qwen3:8b", "qwen3.5:9b"):
            c = self.cell(model, "overthink")
            self.assertFalse(c["paired_usable_only"]["significant"], model)
            self.assertGreater(c["off"]["accuracy"], 0.87, model)
            self.assertGreater(c["on"]["accuracy"], 0.87, model)

    def test_negative_control_the_overthink_set_is_gradeable_at_all(self):
        # A category nobody can score would produce the same null. Both conditions
        # answered nearly everything, so the null is about reasoning.
        for model in ("qwen3:8b", "qwen3.5:9b"):
            c = self.cell(model, "overthink")
            self.assertEqual(c["off"]["usable"], c["off"]["n"], model)
            self.assertGreaterEqual(c["on"]["usable"], 30, model)

    # ------------------------------------------------------ the ceiling caveat

    def test_the_baseline_is_high_enough_to_limit_what_could_be_won(self):
        # README states this as a limitation rather than hiding it.
        for c in self.cells:
            self.assertGreaterEqual(c["off"]["accuracy"], 0.70,
                                    f"{c['model']}/{c['type']} baseline")

    def test_detectable_effect_range_is_the_one_quoted(self):
        widths = [c["paired"]["half_width"] * 100 for c in self.cells]
        nonzero = [w for w in widths if w > 0]
        self.assertAlmostEqual(min(nonzero), 4.9, places=1)
        self.assertAlmostEqual(max(widths), 13.9, places=1)
        # Two cells agree on every item, which is why their half width is zero.
        zero = [c for c in self.cells if c["paired"]["half_width"] == 0]
        self.assertEqual(len(zero), 2)
        for c in zero:
            self.assertEqual(c["flips"]["on_only"], 0)
            self.assertEqual(c["flips"]["off_only"], 0)
            self.assertEqual(c["flips"]["agree"], c["paired"]["n"])

    def test_settings_are_the_ones_the_readme_documents(self):
        meta = self.summary["meta"]
        self.assertEqual(meta["temperature"], 0.0)
        self.assertEqual(meta["top_k"], 1)
        self.assertEqual(meta["num_predict"], 4096)
        self.assertEqual(meta["n_items_per_type"], 40)
        self.assertEqual(len(meta["models"]), 2)


if __name__ == "__main__":
    unittest.main()
