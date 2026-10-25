import unittest

from agent.evaluation.results.evidence_batches.eval_evidence_batches import (
    batches, citation_union, context_for, summarize,
)
from agent.evaluation.results.jev_decision.eval_jev_decision import (
    citation_union as jev_citation_union, decision, final_citations, questions,
    scored, state,
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

    def test_jev_state_has_no_gold_label(self):
        value = state("A claim", [{"id": "Page:2", "text": "A source"}])
        self.assertIn("E1 [Page] A source", value)
        self.assertNotIn("SUPPORTED", value)

    def test_jev_can_choose_its_own_citations_and_score_fever(self):
        shown = [{"id": f"Page:{i}", "text": f"sentence {i}"} for i in range(12)]
        prompt_questions = questions(shown[:10], 0, include_verdict=True)
        self.assertEqual(len(prompt_questions), 11)
        self.assertEqual(prompt_questions["e1"]["type"], "noul")
        self.assertEqual(len(jev_citation_union(shown, [
            {"selected": [item["id"] for item in shown]}])), 12)
        decision = {"pred": "SUPPORTED",
                    "relevance": [{"id": "Page:1", "probability": 0.9},
                                  {"id": "Page:2", "probability": 0.2}],
                    "selected": ["Page:1"]}
        claim = {"label": "SUPPORTS", "evidence": [[[None, None, "Page", 1]]]}
        self.assertEqual(final_citations(decision), ["Page:1"])
        self.assertTrue(scored(claim, decision)["fever_score"])

    def test_jev_decision_batches_questions_against_same_state(self):
        class FakeJev:
            def __init__(self):
                self.states = []

            def predict(self, state, questions):
                self.states.append(state)
                answers = {key: ({"choice": "REFUTED",
                                  "probabilities": {"REFUTED": 0.8}}
                                if key == "verdict" else {"noul": 0.7})
                           for key in questions}
                return {"answers": answers}

        model = FakeJev()
        evidence = [{"id": f"Page:{i}", "text": f"sentence {i}"} for i in range(5)]
        output = decision(model, "A claim", evidence, question_batch_size=2,
                          threshold=0.5)
        self.assertEqual(output["n_calls"], 3)
        self.assertEqual(output["pred"], "REFUTED")
        self.assertEqual(output["selected"], [item["id"] for item in evidence])
        self.assertEqual(len(set(model.states)), 1)


if __name__ == "__main__":
    unittest.main()
