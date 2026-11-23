"""Mechanical output validation and known-action quality probes; no model or shell execution."""

import json
import unittest

from quality_test import diagnostic_cases, diagnostic_pass, parse_response


class QualityTests(unittest.TestCase):
    def answer(self, keys="id\n", duration=0.1, complete=False):
        return dict(
            analysis="Observed current state.",
            plan="Take the requested next action.",
            commands=[dict(keystrokes=keys, duration=duration)],
            task_complete=complete,
        )

    def test_reasoning_and_eos_are_separate_from_json(self):
        value = self.answer()
        parsed = parse_response(
            "<think>Reasoning</think>" + json.dumps(value) + "<|im_end|>"
        )
        self.assertTrue(parsed["schema_valid"])
        self.assertEqual(parsed["parsed"], value)

    def test_truncation_and_incomplete_fences_do_not_crash(self):
        for text in ("<think>unfinished", '{"analysis":', "```bad```"):
            self.assertFalse(parse_response(text)["schema_valid"])

    def test_newline_and_duration_checks(self):
        for value in (
            self.answer(keys="id"),
            self.answer(duration=True),
            self.answer(duration=61),
        ):
            self.assertFalse(parse_response(json.dumps(value))["schema_valid"])
        self.assertTrue(
            parse_response(json.dumps(self.answer(keys="", duration=10)))[
                "schema_valid"
            ]
        )
        self.assertTrue(
            parse_response(json.dumps(self.answer(keys="C-c")))["schema_valid"]
        )
        self.assertFalse(parse_response('{"analysis":NaN}')["json_valid"])

    def test_all_diagnostic_oracles_and_false_completion(self):
        for case in diagnostic_cases():
            expected = case["expected"]
            answer = self.answer()
            answer["commands"] = [
                dict(keystrokes=k, duration=expected.get("duration", 0.1))
                for k in expected["keystrokes"]
            ]
            answer["task_complete"] = expected["task_complete"]
            self.assertTrue(diagnostic_pass(case, parse_response(json.dumps(answer))))
            answer["task_complete"] = not expected["task_complete"]
            self.assertFalse(diagnostic_pass(case, parse_response(json.dumps(answer))))
            answer.pop("task_complete")
            self.assertFalse(diagnostic_pass(case, parse_response(json.dumps(answer))))


if __name__ == "__main__":
    unittest.main()
