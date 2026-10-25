import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.evaluation.results.rsi_jev_evidence_selection_baseline import eval_rsi_jev_evidence_selection_baseline as evaluator
from agent.evaluation.results.rsi_jev_evidence_selection_baseline.eval_rsi_jev_evidence_selection_baseline import (
    KEEP, RETRY, STOP, action_diagnostic, questions_for, select,
    selection_state_for, terminal_verdict,
)


class FakeController:
    def __init__(self):
        self.calls = 0
        self.states = []

    def predict(self, *, state, questions):
        self.calls += 1
        self.states.append(json.loads(state))
        if "verdict" in questions:
            return {"answers": {"verdict": {"choice": "ACCEPT", "confidence": 0.7}}}
        answers = {}
        for key in questions:
            answers[key] = {"choice": RETRY if self.calls == 1 else STOP}
            if key.startswith("keep_"):
                answers[key]["choice"] = KEEP
        return {"answers": answers}


class EvidenceSelectionTest(unittest.TestCase):
    def row(self):
        return {"id": 1, "claim": "A claim",
                "shown": [{"id": "A:1", "text": "first"},
                          {"id": "B:2", "text": "uncited"}],
                "chunks": [{"evidence": [{"id": "A:1", "text": "first"},
                                         {"id": "B:2", "text": "uncited"}],
                            "draws": [
                                {"draw_index": 0, "pred": "SUPPORTED", "format_ok": True,
                                 "token_conf": 0.8, "evidence": ["A:1"]},
                                {"draw_index": 1, "pred": "SUPPORTED", "format_ok": True,
                                 "token_conf": 0.8, "evidence": ["A:1"]},
                                {"draw_index": 2, "pred": "SUPPORTED", "format_ok": True,
                                 "token_conf": 0.8, "evidence": ["A:1"]}]}]}

    def test_keep_persists_across_retry_as_set(self):
        controller = FakeController()
        result = select(self.row(), controller)
        self.assertEqual([item["id"] for item in result["selected_evidence"]], ["A:1"])
        self.assertEqual(result["duplicate_keep_decisions_removed_by_set"], 1)
        self.assertEqual(result["chunk_verifier_calls"], 2)
        self.assertEqual(result["chunks"][0]["attempts"][0]["action"], RETRY)
        self.assertEqual(result["chunks"][0]["attempts"][1]["action"], STOP)
        self.assertEqual(result["chunks"][0]["kept_ids"], ["A:1"])
        self.assertNotIn("uncited", json.dumps(controller.states))
        self.assertEqual(len(controller.states[1]["verifier_output"]["cited_evidence"]), 1)
        self.assertNotIn("observed_verifier_outputs", controller.states[1])

    def test_keep_questions_only_for_verifier_citations(self):
        row = self.row()
        chunk = row["chunks"][0]
        questions = questions_for(chunk, chunk["draws"][0], {"A:1": "E1", "B:2": "E2"})
        self.assertIn("keep_0", questions)
        self.assertNotIn("keep_1", questions)
        self.assertNotIn("Text: uncited", json.dumps(questions))
        self.assertNotIn("Text: first", json.dumps(questions))
        state = json.loads(selection_state_for(row, 0, 0))
        self.assertEqual(state["verifier_output"]["cited_evidence"][0]["text"], "first")
        self.assertNotIn("uncited", json.dumps(state))

    def test_terminal_verdict_is_record_only_and_uses_saved_evidence(self):
        controller = FakeController()
        row = self.row()
        policy = select(row, controller)
        answer = terminal_verdict(row, policy, controller)
        self.assertEqual(answer["fever_label"], "SUPPORTED")
        terminal_state = controller.states[-1]
        self.assertEqual([item["id"] for item in terminal_state["saved_evidence"]], ["A:1"])
        self.assertNotIn("uncited", json.dumps(terminal_state))
        self.assertNotIn("gold", terminal_state)

    def test_action_diagnostic_separates_retry_and_forced_stop(self):
        row = self.row()
        row["chunks"][0]["draws"][0]["correct"] = True
        row["chunks"][0]["draws"][1]["correct"] = False
        result = select(row, FakeController())
        diagnostic = action_diagnostic(row, result)
        self.assertEqual(diagnostic["retry_after_correct_draws"], 1)
        self.assertEqual(diagnostic["correct_retry_then_wrong_selected_chunks"], 1)

    def test_full_run_keeps_chunk_and_pooled_samples_separate(self):
        row = self.row()
        row.update({"gold": "SUPPORTED", "gold_in_context": True})
        class FakeVerifier:
            calls = []
            def __init__(self, *args, **kwargs):
                self.__class__.calls = []
                self.settings = kwargs
            def run(self, prompts, seed):
                self.__class__.calls.append((len(prompts), seed))
                return [{"text": str(i)} for i in range(len(prompts))]
        def fake_score(claim, evidence, generation):
            return {"pred": "SUPPORTED", "gold": "SUPPORTED", "correct": True,
                    "fever_score": True, "evidence_f1": 1.0, "format_ok": True,
                    "token_conf": 0.9, "evidence": [item[1] + ":" + str(item[2]) for item in evidence],
                    "raw": generation["text"]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            output = Path(temp) / "result"
            manifest = {"model": "fake", "adapter": "fake", "prompt_style": "direct",
                        "model_batch_size": 4, "max_new_tokens": 160,
                        "generation_seed": 0}
            fake_torch = types.SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: False, empty_cache=lambda: None))
            with patch.object(evaluator, "validate_source", return_value=(manifest, [row], {1: row})), \
                    patch.object(evaluator, "load_controller", side_effect=lambda *a: FakeController()), \
                    patch.object(evaluator, "LocalModel", FakeVerifier), \
                    patch.object(evaluator, "scored", side_effect=fake_score), \
                    patch.object(evaluator, "gold_sets", return_value=[[('A', 1)]]), \
                    patch.object(evaluator.importlib.metadata, "version", return_value="test"), \
                    patch.dict(sys.modules, {"torch": fake_torch}):
                evaluator.run(SimpleNamespace(source_run=str(source), output_dir=str(output),
                                              trained_head=None, backend="rsi-jev",
                                              model="fake-jev", threshold=0.5))
            pooled = json.loads((output / "pooled_samples.jsonl").read_text().splitlines()[0])
            result = json.loads((output / "predictions.jsonl").read_text().splitlines()[0])
            self.assertEqual(pooled["actual_qwen_calls"], 3)
            self.assertEqual(FakeVerifier.calls, [(3, 100001)])
            self.assertEqual(result["jev_final"]["pred"], "SUPPORTED")
            self.assertEqual(result["jev_terminal_verdict"]["fever_label"], "SUPPORTED")
            self.assertEqual(result["new_final_qwen_calls"], 3)


if __name__ == "__main__":
    unittest.main()
