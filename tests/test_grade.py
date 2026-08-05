"""Grading tests.

Every assertion here is paired with a negative control: a case that must fail.
An assertion that only ever sees data it accepts cannot tell a working grader
from `return True`.
"""

import unittest

from thinktrace.grade import (
    extract_answer,
    grade_choice,
    grade_constraint,
    grade_numeric,
    grade_response,
    grade_text,
    normalize_text,
    parse_number,
    strip_fences,
)


class TestExtraction(unittest.TestCase):
    def test_takes_the_last_answer_line(self):
        text = "Answer: 3\nWait, let me redo that.\nAnswer: 41"
        self.assertEqual(extract_answer(text), "41")

    def test_negative_control_does_not_take_the_first(self):
        text = "Answer: 3\nWait, let me redo that.\nAnswer: 41"
        self.assertNotEqual(extract_answer(text), "3")

    def test_falls_back_to_last_non_empty_line(self):
        self.assertEqual(extract_answer("some reasoning\n\n670\n\n"), "670")

    def test_negative_control_empty_stays_empty(self):
        self.assertEqual(extract_answer("   \n\n  "), "")
        self.assertEqual(extract_answer(""), "")

    def test_handles_bold_markdown_label(self):
        self.assertEqual(extract_answer("**Answer:** Ben"), "Ben")

    def test_strips_code_fences(self):
        self.assertEqual(strip_fences('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_negative_control_leaves_unfenced_text_alone(self):
        self.assertEqual(strip_fences("plain text"), "plain text")


class TestNormalise(unittest.TestCase):
    def test_case_accent_and_punctuation_are_ignored(self):
        self.assertEqual(normalize_text("Brasília."), normalize_text("brasilia"))

    def test_leading_article_is_ignored(self):
        self.assertEqual(normalize_text("The Amazon"), "amazon")

    def test_negative_control_different_words_stay_different(self):
        self.assertNotEqual(normalize_text("Ottawa"), normalize_text("Toronto"))

    def test_parse_number_handles_words_and_separators(self):
        self.assertEqual(parse_number("1,234"), 1234.0)
        self.assertEqual(parse_number("seven"), 7.0)

    def test_negative_control_unparseable_returns_none(self):
        self.assertIsNone(parse_number("no digits here"))


class TestNumericGrader(unittest.TestCase):
    item = {"answer": "670", "accept": [], "grader": "numeric"}

    def test_accepts_the_right_number(self):
        self.assertTrue(grade_numeric(self.item, "670"))
        self.assertTrue(grade_numeric(self.item, "**670**"))

    def test_negative_control_rejects_a_near_miss(self):
        self.assertFalse(grade_numeric(self.item, "671"))
        self.assertFalse(grade_numeric(self.item, "6700"))

    def test_negative_control_rejects_no_number(self):
        self.assertFalse(grade_numeric(self.item, "I am not sure"))


class TestTextGrader(unittest.TestCase):
    item = {"answer": "Canberra", "accept": [], "grader": "text"}

    def test_accepts_exact_and_short_spelled_out(self):
        self.assertTrue(grade_text(self.item, "Canberra"))
        self.assertTrue(grade_text(self.item, "The capital is Canberra."))

    def test_negative_control_rejects_a_different_city(self):
        self.assertFalse(grade_text(self.item, "Sydney"))

    def test_negative_control_rejects_a_long_essay_that_mentions_it(self):
        essay = " ".join(["padding"] * 20) + " Canberra"
        self.assertFalse(grade_text(self.item, essay))

    def test_accept_list_is_honoured(self):
        item = {"answer": "Bern", "accept": ["Berne"], "grader": "text"}
        self.assertTrue(grade_text(item, "Berne"))
        self.assertFalse(grade_text(item, "Zurich"))

    def test_word_boundaries_are_respected(self):
        item = {"answer": "lead", "accept": [], "grader": "text"}
        self.assertFalse(grade_text(item, "leader"))


class TestChoiceGrader(unittest.TestCase):
    options = {
        "feathers": ["feathers"],
        "lead": ["lead"],
        "same": ["same", "equal", "neither"],
    }
    item = {"answer": "feathers", "options": options, "grader": "choice"}

    def test_accepts_the_right_option_in_a_phrase(self):
        self.assertTrue(grade_choice(self.item, "9 kilograms of feathers"))

    def test_negative_control_rejects_the_memorised_answer(self):
        self.assertFalse(grade_choice(self.item, "They weigh the same"))

    def test_ambiguous_reply_naming_two_options_is_wrong(self):
        self.assertFalse(grade_choice(self.item, "feathers and lead"))

    def test_reply_naming_no_option_is_wrong(self):
        self.assertFalse(grade_choice(self.item, "it depends"))


class TestConstraintGrader(unittest.TestCase):
    def test_word_count(self):
        item = {"check": "word_count", "params": {"n": 5}, "grader": "constraint"}
        self.assertTrue(grade_constraint(item, "Rain falls softly on roofs"))
        self.assertFalse(grade_constraint(item, "Rain falls softly on the roofs"))

    def test_repeat_word(self):
        item = {"check": "repeat_word", "params": {"word": "ripple", "n": 3},
                "grader": "constraint"}
        self.assertTrue(grade_constraint(item, "ripple ripple ripple"))
        self.assertFalse(grade_constraint(item, "ripple ripple"))

    def test_upper_csv(self):
        item = {"check": "upper_csv", "params": {"n": 3}, "grader": "constraint"}
        self.assertTrue(grade_constraint(item, "RED, GREEN, BLUE"))
        self.assertFalse(grade_constraint(item, "Red, Green, Blue"))
        self.assertFalse(grade_constraint(item, "RED, GREEN"))

    def test_json_keys_checks_order_and_values(self):
        item = {"check": "json_keys", "params": {"keys": ["a", "b"]},
                "grader": "constraint"}
        self.assertTrue(grade_constraint(item, '{"a": 1, "b": 1}'))
        self.assertFalse(grade_constraint(item, '{"b": 1, "a": 1}'))
        self.assertFalse(grade_constraint(item, '{"a": 1, "b": 2}'))
        self.assertFalse(grade_constraint(item, "not json"))

    def test_exact_chars(self):
        item = {"check": "exact_chars", "params": {"char": "*", "n": 4},
                "grader": "constraint"}
        self.assertTrue(grade_constraint(item, "****"))
        self.assertFalse(grade_constraint(item, "*****"))

    def test_desc_range(self):
        item = {"check": "desc_range", "params": {"lo": 3, "hi": 6},
                "grader": "constraint"}
        self.assertTrue(grade_constraint(item, "6,5,4,3"))
        self.assertFalse(grade_constraint(item, "3,4,5,6"))

    def test_start_end(self):
        item = {"check": "start_end", "params": {"start": "Blue", "end": "today"},
                "grader": "constraint"}
        self.assertTrue(grade_constraint(item, "Blue skies arrived today."))
        self.assertFalse(grade_constraint(item, "The blue skies arrived today."))

    def test_forbidden_letter(self):
        item = {"check": "forbidden_letter", "params": {"letter": "e", "min_words": 5},
                "grader": "constraint"}
        self.assertTrue(grade_constraint(item, "A big storm rolls south"))
        self.assertFalse(grade_constraint(item, "A big storm rolls over there"))
        self.assertFalse(grade_constraint(item, "Storm today"))

    def test_unknown_check_raises_rather_than_passing(self):
        item = {"check": "nope", "params": {}, "grader": "constraint"}
        with self.assertRaises(ValueError):
            grade_constraint(item, "anything")


class TestUsability(unittest.TestCase):
    item = {"answer": "5", "accept": [], "grader": "numeric"}

    def test_empty_content_is_unusable_not_wrong(self):
        r = grade_response(self.item, "", "length")
        self.assertFalse(r["usable"])
        self.assertFalse(r["correct"])

    def test_negative_control_non_empty_content_is_usable(self):
        r = grade_response(self.item, "Answer: 5", "stop")
        self.assertTrue(r["usable"])
        self.assertTrue(r["correct"])

    def test_wrong_but_present_is_usable(self):
        r = grade_response(self.item, "Answer: 6", "stop")
        self.assertTrue(r["usable"])
        self.assertFalse(r["correct"])


if __name__ == "__main__":
    unittest.main()
