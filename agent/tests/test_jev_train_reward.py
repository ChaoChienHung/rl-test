import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from agent.train.train_jev_rl import pooled_decision_reward, sample_pooled_episode
from agent.train.replay_env import RETRY, STOP, stop_retry_feedback, verifier_result_good


class JevTrainRewardTest(unittest.TestCase):
    def test_pooled_action_reward_and_extra_draw_cost(self):
        good = {"gold": "SUPPORTED", "correct": True, "evidence_f1": 0.8}
        bad = {"gold": "SUPPORTED", "correct": False, "evidence_f1": 0.9}
        pooled = {"draws": [bad, good, bad], "selected": good,
                  "decisions": [{"draw_index": 0, "action": RETRY},
                                {"draw_index": 1, "action": STOP}],
                  "extra_draws": 1}
        reward, detail = pooled_decision_reward(pooled, chunk_extra_draws=2)
        self.assertEqual(detail["pooled_action_feedback"], [1.0, 1.0])
        self.assertAlmostEqual(reward, 1.85)

    def test_bad_stop_and_good_retry_are_penalized(self):
        good = {"gold": "SUPPORTED", "correct": True, "evidence_f1": 0.8}
        bad = {"gold": "SUPPORTED", "correct": False, "evidence_f1": 0.8}
        self.assertEqual(stop_retry_feedback(STOP, bad), -1)
        self.assertEqual(stop_retry_feedback(RETRY, good), -1)
        pooled = {"draws": [bad, good, bad], "selected": bad,
                  "decisions": [{"draw_index": 0, "action": STOP}],
                  "extra_draws": 0}
        reward, detail = pooled_decision_reward(pooled, chunk_extra_draws=0)
        self.assertEqual(detail["final_good"], False)
        self.assertEqual(reward, -2)

    def test_f1_boundary_and_nei(self):
        self.assertFalse(verifier_result_good(
            {"gold": "SUPPORTED", "correct": True, "evidence_f1": 2 / 3}))
        self.assertTrue(verifier_result_good(
            {"gold": "NOT_ENOUGH_INFO", "correct": True, "evidence_f1": 0}))

    def test_training_pooled_stage_matches_forced_third_vote(self):
        row = {"claim": "claim"}
        claim = {"id": 1, "label": "SUPPORTS"}
        episode = {"selected_evidence": [{"id": "A:1", "text": "first"}]}
        class FakeVerifier:
            def run(self, prompts, seed):
                self.calls = (len(prompts), seed)
                return [{"index": i} for i in range(len(prompts))]
        verifier = FakeVerifier()
        policy = SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext))
        draws = [
            {"pred": "SUPPORTED", "token_conf": 0.2, "correct": True, "evidence": ["A:1"]},
            {"pred": "REFUTED", "token_conf": 0.3, "correct": False, "evidence": []},
            {"pred": "SUPPORTED", "token_conf": 0.1, "correct": True, "evidence": ["A:1"]},
        ]
        with patch("agent.train.train_jev_rl.scored", side_effect=lambda c, e, g: draws[g["index"]]), \
                patch("agent.train.train_jev_rl.sample_question",
                      side_effect=[("RETRY", 0.0, 0.0, 0.0),
                                   ("RETRY", 0.0, 0.0, 0.0)]):
            result = sample_pooled_episode(row, claim, episode, verifier,
                                           policy, None, 0.0, 100001)
        self.assertEqual(verifier.calls, (3, 100001))
        self.assertEqual(result["extra_draws"], 2)
        self.assertEqual(result["selected_draw_index"], 0)


if __name__ == "__main__":
    unittest.main()
