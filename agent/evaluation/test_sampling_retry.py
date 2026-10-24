import unittest

from agent.evaluation.eval_sampling_retry import stopping_index, summarize


def sample(pred, gold="REFUTED", confidence=0.8, evidence=(), format_ok=True):
    return {
        "pred": pred, "gold": gold, "correct": pred == gold,
        "fever_score": pred == gold and (gold == "NOT_ENOUGH_INFO" or bool(evidence)),
        "evidence_f1": 1.0 if evidence else 0.0 if gold != "NOT_ENOUGH_INFO" else None,
        "format_ok": format_ok, "token_conf": confidence,
        "evidence": list(evidence), "in_tokens": 100, "out_tokens": 10,
    }


class SamplingRetryTest(unittest.TestCase):
    def test_nei_low_confidence_and_bad_format_retry(self):
        draws = [sample("NOT_ENOUGH_INFO", confidence=0.9),
                 sample("SUPPORTED", confidence=0.4),
                 sample("REFUTED", evidence=("Page:1",))]
        self.assertEqual(stopping_index(draws, 0.5), 2)
        draws[0] = sample("REFUTED", evidence=("Page:1",), format_ok=False)
        self.assertEqual(stopping_index(draws, 0.5), 2)

    def test_early_stop_and_costs(self):
        draws = [sample("SUPPORTED"), sample("REFUTED", evidence=("Page:1",)),
                 sample("REFUTED", evidence=("Page:1",))]
        self.assertEqual(stopping_index(draws, 0.5), 0)
        rows = [{"samples": draws, "gold": "REFUTED", "gold_in_context": True,
                 "token_conf_std": 0.0,
                 "heuristic": {"selected_index": 0, "n_verify": 1}}]
        result = summarize(rows)
        self.assertEqual(result["overall"]["heuristic"]["accuracy"], 0.0)
        self.assertEqual(result["overall"]["third_draw"]["accuracy"], 1.0)
        self.assertEqual(result["early_wrong_with_later_correct"], 1)
        self.assertEqual(result["actual_verifier_calls"], 3)
        self.assertEqual(result["mean_simulated_verifier_calls"], 1.0)
        self.assertEqual(result["gold_in_context"]["n"], 1)
        self.assertEqual(result["gold_not_in_context"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
