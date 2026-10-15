"""Check that native thinking is trained with the intended final-answer reward."""

import unittest

from verifier.experiments._shared.grpo_cot import (
    load_train_rows, make_reward_funcs, score_one, thinking_present,
)
from verifier.model import parse


class NativeThinkingRewardTests(unittest.TestCase):
    def test_only_final_answer_is_parsed(self):
        text = '<think>The draft was {"verdict":"REFUTED"}. E1 settles it.</think>\n'
        text += '{"verdict":"SUPPORTED","evidence":["E1"]}'
        obj, ok = parse(text)
        self.assertEqual(obj["verdict"], "SUPPORTED")
        self.assertTrue(all(ok[key] for key in ("json", "verdict", "evidence")))
        self.assertEqual(score_one(text, "SUPPORTED", ["E1"])[0], 1.5)

    def test_no_confidence_reward_and_format_failure(self):
        text = '<think>E1 supports the claim.</think>{"verdict":"SUPPORTED","evidence":["E1"]}'
        with_conf = text[:-1] + ',"confidence":0.0}'
        self.assertEqual(score_one(text, "SUPPORTED", ["E1"])[0],
                         score_one(with_conf, "SUPPORTED", ["E1"])[0])
        self.assertEqual(score_one('<think>Still reasoning', "SUPPORTED", ["E1"])[0], -2.0)

    def test_reasoning_metric_is_not_part_of_reward(self):
        with_thought = '<think>Evidence E1 supports it.</think>{"verdict":"SUPPORTED","evidence":["E1"]}'
        without_thought = '{"verdict":"SUPPORTED","evidence":["E1"]}'
        funcs = make_reward_funcs()
        kwargs = {"gold": ["SUPPORTED", "SUPPORTED"], "gold_ids": [["E1"], ["E1"]]}
        completions = [[{"content": with_thought}], [{"content": without_thought}]]
        self.assertEqual(funcs[0](completions, **kwargs), [1.5, 1.5])
        self.assertEqual(thinking_present(completions), [1.0, 0.0])

    def test_fixed_training_split(self):
        rows = load_train_rows("verifier/data/train/sft/fever-oracle-matched-grpo-1200", 1200)
        self.assertEqual(len(rows), 1200)
        self.assertEqual(sum(row["gold"] == "NOT_ENOUGH_INFO" for row in rows), 407)
        self.assertTrue(all(row["gold_ids"] is not None for row in rows))


if __name__ == "__main__":
    unittest.main()
