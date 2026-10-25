import json
import unittest

from agent.evaluation.pooled_retry import choose_pooled, state_for_pool


class FakeJev:
    def __init__(self):
        self.states = []

    def predict(self, *, state, questions):
        self.states.append(json.loads(state))
        return {"answers": {"action": {"choice": "RETRY" if len(self.states) == 1 else "STOP"}}}


class PooledRetryTest(unittest.TestCase):
    def test_same_evidence_three_draws_and_current_output_only(self):
        evidence = [{"id": "A:1", "text": "cited"},
                    {"id": "B:2", "text": "hidden"}]
        draws = [{"pred": "NOT_ENOUGH_INFO", "format_ok": True,
                  "token_conf": 0.2, "evidence": [], "correct": False},
                 {"pred": "SUPPORTED", "format_ok": True,
                  "token_conf": 0.9, "evidence": ["A:1"], "correct": True},
                 {"pred": "REFUTED", "format_ok": True,
                  "token_conf": 0.8, "evidence": ["B:2"], "correct": False}]
        model = FakeJev()
        result = choose_pooled("claim", evidence, draws, model)
        self.assertEqual(result["selected_draw_index"], 1)
        self.assertEqual(result["simulated_qwen_calls"], 2)
        self.assertEqual(result["controller_calls"], 2)
        self.assertEqual(model.states[0]["verifier_output"]["cited_evidence"], [])
        self.assertEqual(model.states[1]["verifier_output"]["cited_evidence"][0]["text"], "cited")
        self.assertNotIn("hidden", json.dumps(model.states))
        self.assertNotIn("gold", json.dumps(model.states))

    def test_heuristic_forces_third_draw_when_confidence_low(self):
        draws = [{"pred": "NOT_ENOUGH_INFO", "format_ok": True,
                  "token_conf": 0.1, "evidence": [], "correct": False} for _ in range(3)]
        result = choose_pooled("claim", [], draws)
        self.assertEqual(result["selected_draw_index"], 0)
        self.assertEqual(result["stopped_at_draw_index"], 2)
        self.assertEqual(result["simulated_qwen_calls"], 3)
        self.assertEqual(result["forced_stop_on_wrong"], 1)

    def test_heuristic_threshold_stops_early_then_votes_over_observed_draws(self):
        draws = [{"pred": "NOT_ENOUGH_INFO", "format_ok": True,
                  "token_conf": 0.2, "evidence": [], "correct": False},
                 {"pred": "SUPPORTED", "format_ok": True,
                  "token_conf": 0.9, "evidence": ["A:1"], "correct": True},
                 {"pred": "REFUTED", "format_ok": True,
                  "token_conf": 0.99, "evidence": [], "correct": False}]
        result = choose_pooled("claim", [{"id": "A:1", "text": "a"}], draws)
        self.assertEqual(result["simulated_qwen_calls"], 2)
        self.assertEqual(result["selected_draw_index"], 1)

    def test_heuristic_majority_after_three_draws(self):
        draws = [{"pred": "SUPPORTED", "format_ok": True,
                  "token_conf": 0.2, "evidence": [], "correct": True},
                 {"pred": "REFUTED", "format_ok": True,
                  "token_conf": 0.3, "evidence": [], "correct": False},
                 {"pred": "SUPPORTED", "format_ok": True,
                  "token_conf": 0.1, "evidence": [], "correct": True}]
        result = choose_pooled("claim", [], draws)
        self.assertEqual(result["simulated_qwen_calls"], 3)
        self.assertEqual(result["selected_draw_index"], 0)
        self.assertEqual(result["stopped_at_draw_index"], 2)


if __name__ == "__main__":
    unittest.main()
