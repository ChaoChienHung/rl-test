"""Checks that the reasoning baseline preserves verdict/evidence scoring."""

import unittest

from stage0 import REASONING_SYSTEM, score


class ReasoningScoreTest(unittest.TestCase):
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
