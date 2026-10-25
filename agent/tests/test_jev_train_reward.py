import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from agent.train.train_jev_rl import pooled_decision_reward, sample_episode, sample_pooled_episode
from agent.train.replay_env import (RETRY, STOP, best_gold_set_evidence_credit,
                                    citation_choice_feedback,
                                    verdict_retry_feedback)


class JevTrainRewardTest(unittest.TestCase):
    def test_citation_variant_discards_retried_draw_citations(self):
        row = {"claim": "claim", "shown": [{"id": "A:1", "text": "first"},
                                           {"id": "B:2", "text": "second"}],
               "chunks": [{"evidence": [{"id": "A:1", "text": "first"},
                                       {"id": "B:2", "text": "second"}],
                           "draws": [{"format_ok": True, "evidence": ["A:1"]},
                                     {"format_ok": True, "evidence": ["B:2"]},
                                     {"format_ok": True, "evidence": []}]}]}
        policy = SimpleNamespace()
        with patch("agent.train.train_jev_rl.sample_question",
                   side_effect=[(RETRY, 0.0, 0.0, 0.0),
                                (STOP, 0.0, 0.0, 0.0)]) as question:
            episode = sample_episode(row, policy, None, "citation")
        self.assertEqual([item["id"] for item in episode["selected_evidence"]],
                         ["B:2"])
        self.assertEqual(episode["trace"][0]["kept_ids"], [])
        self.assertEqual(episode["extra_draws"], 1)
        self.assertIn("RETRY discards", question.call_args_list[0].args[4]["instructions"])

    def test_keep_drop_feedback_is_asymmetric(self):
        gold = {"A:1"}
        self.assertEqual(citation_choice_feedback("KEEP", "A:1", gold), 1.0)
        self.assertEqual(citation_choice_feedback("DROP", "A:1", gold), -0.5)
        self.assertEqual(citation_choice_feedback("KEEP", "B:2", gold), -0.25)
        self.assertEqual(citation_choice_feedback("DROP", "B:2", gold), 0.1)

    def test_selection_does_not_ask_keep_on_retry(self):
        row = {"claim": "claim", "shown": [{"id": "A:1", "text": "first"},
                                           {"id": "B:2", "text": "second"}],
               "chunks": [{"evidence": [{"id": "A:1", "text": "first"},
                                       {"id": "B:2", "text": "second"}],
                           "draws": [{"format_ok": True, "evidence": ["A:1"]},
                                     {"format_ok": True, "evidence": ["B:2"]},
                                     {"format_ok": True, "evidence": []}]}]}
        with patch("agent.train.train_jev_rl.sample_question",
                   side_effect=[(RETRY, 0.0, 0.0, 0.0),
                                (STOP, 0.0, 0.0, 0.0),
                                ("KEEP", 0.0, 0.0, 0.0)]) as question:
            episode = sample_episode(row, SimpleNamespace(), None, "selection")
        self.assertEqual(question.call_count, 3)
        self.assertEqual(episode["trace"][0]["kept_ids"], [])
        self.assertEqual(episode["citation_decisions"],
                         [{"id": "B:2", "choice": "KEEP"}])
        self.assertEqual([item["id"] for item in episode["selected_evidence"]], ["B:2"])

    def test_citation_action_reward_uses_stop_draw_choices(self):
        claim = {"id": 1, "label": "SUPPORTS"}
        good = {"gold": "SUPPORTED", "correct": True, "evidence": ["A:1"]}
        pooled = {"draws": [good] * 3, "selected": good,
                  "decisions": [{"draw_index": 0, "action": STOP}],
                  "extra_draws": 0}
        with patch("agent.train.train_jev_rl.gold_sets", return_value=[[('A', 1)]]):
            base, _ = pooled_decision_reward(pooled, claim, 0)
            improved, detail = pooled_decision_reward(
                pooled, claim, 0,
                citation_decisions=[{"id": "A:1", "choice": "KEEP"},
                                    {"id": "B:2", "choice": "DROP"}])
        self.assertEqual(detail["citation_action_feedback"], [1.0, 0.1])
        self.assertAlmostEqual(improved - base, 0.25 * 0.55)

    def test_pooled_action_reward_and_extra_draw_cost(self):
        claim = {"id": 1, "label": "SUPPORTS"}
        good = {"gold": "SUPPORTED", "correct": True, "evidence": ["A:1"]}
        bad = {"gold": "SUPPORTED", "correct": False, "evidence": []}
        pooled = {"draws": [bad, good, bad], "selected": good,
                  "decisions": [{"draw_index": 0, "action": RETRY},
                                {"draw_index": 1, "action": STOP}],
                  "extra_draws": 1}
        with patch("agent.train.train_jev_rl.gold_sets", return_value=[[('A', 1)]]):
            reward, detail = pooled_decision_reward(pooled, claim, chunk_extra_draws=2)
        self.assertEqual(detail["pooled_action_feedback"], [0.25, 1.0])
        self.assertEqual(detail["evidence_detail"]["complete_gold_set"], True)
        self.assertAlmostEqual(reward, 2.1)

    def test_bad_stop_and_good_retry_are_penalized(self):
        claim = {"id": 1, "label": "SUPPORTS"}
        good = {"gold": "SUPPORTED", "correct": True, "evidence": ["A:1"]}
        bad = {"gold": "SUPPORTED", "correct": False, "evidence": []}
        self.assertEqual(verdict_retry_feedback(STOP, bad, bad), -1)
        self.assertEqual(verdict_retry_feedback(RETRY, good, bad), -1.5)
        self.assertEqual(verdict_retry_feedback(RETRY, bad, bad), -1)
        pooled = {"draws": [bad, good, bad], "selected": bad,
                  "decisions": [{"draw_index": 0, "action": STOP}],
                  "extra_draws": 0}
        with patch("agent.train.train_jev_rl.gold_sets", return_value=[[('A', 1)]]):
            reward, detail = pooled_decision_reward(pooled, claim, chunk_extra_draws=0)
        self.assertEqual(detail["final_verdict_correct"], False)
        self.assertEqual(reward, -2)

    def test_partial_best_gold_set_and_off_gold_penalty(self):
        groups = [{"A:1", "A:2", "A:3"}, {"B:1"}]
        exact, detail = best_gold_set_evidence_credit("SUPPORTED", ["B:1"], groups)
        self.assertEqual(exact, 1.25)
        self.assertTrue(detail["complete_gold_set"])
        partial, detail = best_gold_set_evidence_credit(
            "SUPPORTED", ["A:1", "A:2", "X:9"], groups)
        self.assertAlmostEqual(partial, 2 / 3 - 0.25 / 3)
        self.assertFalse(detail["complete_gold_set"])
        self.assertEqual(detail["off_gold_count"], 1)

    def test_nei_empty_evidence(self):
        self.assertEqual(best_gold_set_evidence_credit(
            "NOT_ENOUGH_INFO", [], [])[0], 0.25)
        self.assertEqual(best_gold_set_evidence_credit(
            "NOT_ENOUGH_INFO", ["A:1"], [])[0], -0.25)

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
