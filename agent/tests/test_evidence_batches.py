import unittest
from unittest.mock import patch

from agent.evaluation.results.evidence_batches.eval_evidence_batches import (
    batches, citation_union, context_for, summarize,
)
from agent.evaluation.results.jev_controller.eval_jev_controller import (
    FULL, UNION, RSIAdapter, choose_route, controller_state, heuristic_route,
    route_evidence, verify_routes,
    summarize as summarize_controller,
)


def result(gold, pred, cited=(), fever=False):
    return {"gold": gold, "pred": pred, "correct": gold == pred,
            "fever_score": fever, "evidence_f1": 1.0 if fever else 0.0,
            "format_ok": True, "in_tokens": 10, "out_tokens": 2,
            "cited": list(cited)}


class EvidenceBatchesTest(unittest.TestCase):
    def test_batch_ids_are_local_and_final_union_can_exceed_batch_size(self):
        shown = [(f"E{i + 1}", "Page", i, f"sentence {i}") for i in range(25)]
        parts = list(batches(shown, 10))
        self.assertEqual([len(part) for part in parts], [10, 10, 5])
        self.assertEqual(parts[1][0][0], "E1")
        outputs = [result("SUPPORTED", "SUPPORTED", cited=[("Page", i) for i in range(10)]),
                   result("SUPPORTED", "SUPPORTED", cited=[("Page", i) for i in range(10, 20)]),
                   result("SUPPORTED", "NOT_ENOUGH_INFO")]
        union = citation_union(shown, outputs)
        self.assertEqual(len(union), 20)
        self.assertEqual(union[-1][0], "E20")
        self.assertEqual(union[-1][2], 19)

    def test_invalid_batch_citations_do_not_enter_union(self):
        shown = [("E1", "Page", 0, "a"), ("E2", "Page", 1, "b")]
        valid = result("REFUTED", "REFUTED", cited=[("Page", 1)])
        invalid = result("REFUTED", "SUPPORTED", cited=[("Page", 0)])
        invalid["format_ok"] = False
        self.assertEqual(citation_union(shown, [invalid, valid]), [("E1", "Page", 1, "b")])

    def test_gold_coverage_and_costs(self):
        claim = {"label": "SUPPORTS", "evidence": [[[None, None, "Page", 1]]]}
        shown = [("E1", "Page", 1, "proof")]
        self.assertTrue(context_for(claim, shown)["gold_in_context"])
        self.assertFalse(context_for(claim, [])["gold_in_context"])
        row = {"gold": "SUPPORTED", "gold_in_context": True,
               "gold_in_union": True, "n_union": 1,
               "baseline": result("SUPPORTED", "SUPPORTED", fever=True),
               "batches": [result("SUPPORTED", "SUPPORTED", fever=True)],
               "final": result("SUPPORTED", "REFUTED")}
        metrics = summarize([row])
        self.assertEqual(metrics["overall"]["accuracy_lost"], 1)
        self.assertEqual(metrics["batched_calls"], 2)
        self.assertEqual(metrics["tokens"]["evidence_batches"]["in_tokens"], 20)

    def test_jev_controller_state_excludes_gold_and_raw_evidence(self):
        row = {"claim": "A claim", "gold": "REFUTED", "n_shown": 2,
               "n_union": 1, "shown": [{"text": "SECRET_EVIDENCE"}],
               "final": {"raw": "SECRET_FINAL"},
               "batches": [{"pred": "SUPPORTED", "format_ok": True,
                            "token_conf": 0.8, "stated_conf": 0.9,
                            "evidence": ["Page:2"]}]}
        value = controller_state(row)
        self.assertIn("Page:2", value)
        self.assertNotIn("REFUTED", value)
        self.assertNotIn("SECRET_EVIDENCE", value)
        self.assertNotIn("SECRET_FINAL", value)

    def test_jev_controller_chooses_route_and_empty_union_falls_back(self):
        class FakeJev:
            def __init__(self):
                self.calls = 0

            def predict(self, state, questions):
                self.calls += 1
                self.assert_state = state
                return {"answers": {"route": {"choice": UNION,
                                               "confidence": 0.4,
                                               "probabilities": {UNION: 0.7, FULL: 0.3}}}}

        model = FakeJev()
        row = {"claim": "A claim", "n_shown": 1, "n_union": 1,
               "batches": [{"pred": "SUPPORTED", "format_ok": True,
                            "token_conf": 0.8, "stated_conf": 0.9,
                            "evidence": ["Page:2"]}],
               "shown": [{"id": "Page:1", "text": "full evidence"}],
               "union": [{"id": "Page:2", "text": "cited evidence"}]}
        choice = choose_route(model, row)
        self.assertEqual(choice["route"], UNION)
        self.assertEqual(choice["selected_probability"], 0.7)
        self.assertEqual(choice["confidence"], 0.4)
        self.assertEqual(model.calls, 1)
        row["n_union"] = 0
        self.assertEqual(choose_route(model, row)["route"], FULL)
        self.assertEqual(model.calls, 1)

    def test_final_qwen_runs_only_after_routes_and_shares_matching_route(self):
        class FakeVerifier:
            def __init__(self):
                self.prompts = []

            def run(self, prompts, seed):
                self.prompts.extend(prompts)
                return [{"raw": "answer", "seed": seed}]

        row = {"claim": "A claim", "shown": [{"id": "Page:1", "text": "full"}],
               "union": [{"id": "Page:2", "text": "cited"}]}
        self.assertEqual(route_evidence(row, UNION), [("E1", "Page", 2, "cited")])
        verifier = FakeVerifier()
        with patch("agent.evaluation.results.jev_controller.eval_jev_controller.scored",
                   side_effect=lambda claim, evidence, gen: {"used": evidence}) as scorer:
            outputs, calls = verify_routes(verifier, {}, row,
                                           {"controller": UNION, "heuristic": UNION}, 17)
        self.assertEqual(calls, 1)
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(outputs["controller"], outputs["heuristic"])
        verifier = FakeVerifier()
        with patch("agent.evaluation.results.jev_controller.eval_jev_controller.scored",
                   side_effect=lambda claim, evidence, gen: {"used": evidence}):
            outputs, calls = verify_routes(verifier, {}, row,
                                           {"controller": UNION, "heuristic": FULL}, 17)
        self.assertEqual(calls, 2)
        self.assertNotEqual(outputs["controller"], outputs["heuristic"])

    def test_rsi_adapter_normalizes_decider_answer_shape(self):
        class FakeDecider:
            def decide(self, state, questions):
                self.last_request = (state, questions)
                return {"route": {"type": "choice", "choice": FULL,
                                  "probabilities": {FULL: 0.6, UNION: 0.4}}}

        decider = FakeDecider()
        answer = RSIAdapter(decider).predict(state="claim summary", questions={"route": {}})
        self.assertEqual(answer["answers"]["route"]["choice"], FULL)
        self.assertEqual(decider.last_request, ("claim summary", {"route": {}}))

    def test_heuristic_routes_from_batch_summaries(self):
        batch = {"pred": "SUPPORTED", "format_ok": True, "token_conf": 0.8}
        row = {"n_union": 1, "batches": [batch, dict(batch)]}
        self.assertEqual(heuristic_route(row)["route"], UNION)
        row["batches"][1] = dict(batch, pred="REFUTED")
        self.assertEqual(heuristic_route(row)["reason"], "batch_verdict_disagreement")
        row["batches"][1] = dict(batch, token_conf=0.4)
        self.assertEqual(heuristic_route(row)["reason"], "low_or_missing_token_confidence")
        row["batches"][1] = dict(batch, pred="NOT_ENOUGH_INFO")
        self.assertEqual(heuristic_route(row)["reason"], "nei_batch")
        row["n_union"] = 0
        self.assertEqual(heuristic_route(row)["reason"], "empty_union")

    def test_jev_controller_summary_compares_new_final_scores(self):
        baseline = result("SUPPORTED", "SUPPORTED", fever=True)
        union = result("SUPPORTED", "REFUTED")
        row = {"gold": "SUPPORTED", "gold_in_context": True,
               "controller_final": baseline, "heuristic_final": union,
               "batches": [union], "n_final_qwen_calls": 2,
               "controller": {"route": FULL, "reason": "openjev",
                              "model_calls": 1, "seconds": 0.1},
               "heuristic": {"route": UNION, "reason": "consistent_confident_batches"}}
        metrics = summarize_controller([row])
        self.assertEqual(metrics["overall"]["jev_vs_heuristic_accuracy_gained"], 1)
        self.assertEqual(metrics["overall"]["jev_vs_heuristic_fever_gained"], 1)
        self.assertEqual(metrics["route_agreement"], 0)
        self.assertEqual(metrics["jev_model_calls"], 1)
        self.assertEqual(metrics["new_final_qwen_calls"], 2)


if __name__ == "__main__":
    unittest.main()
