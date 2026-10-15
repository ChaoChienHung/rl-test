"""Checks that the reasoning baseline preserves verdict/evidence scoring."""

import unittest

from verifier.model import COT_SFT_SYSTEM, REASONING_SYSTEM, adapter_label, score


class ReasoningScoreTest(unittest.TestCase):
    def test_experiment_adapter_label_keeps_run_identity(self):
        path = "../experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run2/artifacts/checkpoints"
        self.assertEqual(adapter_label(path), "sft-direct-from-base-qwen3p5-2b-5895-samples_run2")
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

    def test_cot_sft_format_does_not_require_unsupported_confidence_label(self):
        claim = {"id": 7, "claim": "Ada wrote a book.", "label": "SUPPORTS",
                 "evidence": [[[0, 0, "Ada", 0]]]}
        context = {"shown": [("E1", "Ada", 0, "Ada wrote a book.")],
                   "gold_page_retrieved": True, "gold_in_context": True}
        generation = {"text": '{"reasoning":"E1 directly supports the claim.",'
                              '"verdict":"SUPPORTED","evidence":["E1"]}',
                      "tokens": [], "provider": "test", "in_tokens": 10, "out_tokens": 30}
        result = score(claim, context, generation, require_confidence=False)
        self.assertTrue(result["format_ok"])
        self.assertFalse(result["confidence_required"])
        self.assertIsNone(result["stated_conf"])
        self.assertFalse(score(claim, context, generation)["format_ok"])
        self.assertNotIn('"confidence"', COT_SFT_SYSTEM)


if __name__ == "__main__":
    unittest.main()
