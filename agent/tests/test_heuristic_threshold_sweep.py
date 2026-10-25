import unittest
from unittest.mock import patch

from agent.evaluation.results.heuristic_threshold_sweep.eval_heuristic_threshold_sweep import (
    generate_pools, summarize,
)


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def run(self, prompts, seed):
        self.calls.append((prompts, seed))
        return [{"draw": index} for index in range(len(prompts))]


class HeuristicThresholdSweepTest(unittest.TestCase):
    def test_pooled_samples_shared_only_for_identical_claim_pool(self):
        row = {"id": 7, "claim": "A claim"}
        evidence_a = [{"id": "A:1", "text": "A"}]
        evidence_b = [{"id": "B:2", "text": "B"}]
        choices = {"low": {"selected_evidence": evidence_a},
                   "same": {"selected_evidence": evidence_a},
                   "high": {"selected_evidence": evidence_b}}
        verifier = FakeVerifier()
        with patch("agent.evaluation.results.heuristic_threshold_sweep.eval_heuristic_threshold_sweep.scored",
                   side_effect=lambda claim, evidence, generation: generation):
            result = generate_pools(row, choices, {}, verifier, {"generation_seed": 0})
        self.assertEqual(result["actual_qwen_calls"], 6)
        self.assertEqual(len(verifier.calls), 2)
        self.assertIs(result["pools"]["low"]["draws"], result["pools"]["same"]["draws"])
        self.assertIsNot(result["pools"]["low"]["draws"], result["pools"]["high"]["draws"])
        self.assertEqual(verifier.calls[0][1], verifier.calls[1][1])

    def test_empty_summary(self):
        result = summarize([])
        self.assertEqual(result["n_claims"], 0)
        self.assertEqual(result["actual_new_pooled_qwen_calls"], 0)


if __name__ == "__main__":
    unittest.main()
