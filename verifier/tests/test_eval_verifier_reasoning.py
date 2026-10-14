"""Checks that the reasoning baseline preserves verdict/evidence scoring."""

import unittest

from verifier.model import REASONING_SYSTEM, adapter_label, score


class ReasoningScoreTest(unittest.TestCase):
    def test_experiment_adapter_label_keeps_run_identity(self):
        path = "../experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints"
        self.assertEqual(adapter_label(path), "sft_qwen_3.5-2b_n6000_run2")
        self.assertEqual(adapter_label("verifier/data/train/sft/fever-oracle-full"),
                         "fever-oracle-full")

    def test_reasoning_output_keeps_existing_verdict_metrics(self):
        claim = {"id": 7, "claim": "Ada wrote a book.", "label": "SUPPORTS",
                 "evidence": [[[0, 0, "Ada", 0]]]}
        context = {"shown": [("E1", "Ada", 0, "Ada wrote a book.")],
                   "gold_page_retrieved": True, "gold_in_context": True}
        generation = {"text": '{"reasoning":"E1 supports the claim.","verdict":"SUPPORTED",'
                              '"evidence":["E1"],"confidence":0.8}',
                      "tokens": [], "provider": "test", "in_tokens": 10, "out_tokens": 30}
        result = score(claim, context, generation)
        self.assertTrue(result["correct"])
        self.assertTrue(result["fever_score"])
        self.assertEqual(result["cited"], [("Ada", 0)])
        self.assertEqual(result["reasoning_cited_ids"], ["E1"])
        self.assertTrue(result["reasoning_citations_valid"])
        self.assertIn('"reasoning"', REASONING_SYSTEM)

    def test_unshown_reasoning_citation_is_flagged(self):
        claim = {"id": 7, "claim": "Ada wrote a book.", "label": "SUPPORTS",
                 "evidence": [[[0, 0, "Ada", 0]]]}
        context = {"shown": [("E1", "Ada", 0, "Ada wrote a book.")],
                   "gold_page_retrieved": True, "gold_in_context": True}
        generation = {"text": '{"reasoning":"E9 proves it.","verdict":"SUPPORTED",'
                              '"evidence":["E1"],"confidence":0.8}',
                      "tokens": [], "provider": "test", "in_tokens": 10, "out_tokens": 30}
        self.assertFalse(score(claim, context, generation)["reasoning_citations_valid"])


if __name__ == "__main__":
    unittest.main()
