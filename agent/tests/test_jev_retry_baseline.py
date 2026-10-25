import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.evaluation.results.rsi_jev_retry_baseline import eval_rsi_jev_retry_baseline as evaluator
from agent.evaluation.results.rsi_jev_retry_baseline.eval_rsi_jev_retry_baseline import (
    QUESTION, RETRY, STOP, heuristic_action, select, state_for,
)


class FakeController:
    def __init__(self):
        self.states = []

    def predict(self, *, state, questions):
        self.states.append(json.loads(state))
        assert questions == QUESTION
        return {"answers": {"action": {"choice": RETRY if len(self.states) == 1 else STOP}}}


class RetryBaselineTest(unittest.TestCase):
    def row(self):
        return {
            "id": 1, "claim": "A claim", "gold": "SUPPORTED",
            "shown": [{"id": "A:1", "text": "first"}, {"id": "B:2", "text": "second"}],
            "chunks": [
                {"evidence": [{"id": "A:1", "text": "first"}], "draws": [
                    {"draw_index": 0, "pred": "NOT_ENOUGH_INFO", "format_ok": True,
                     "token_conf": 0.8, "evidence": [], "gold": "SUPPORTED", "correct": False},
                    {"draw_index": 1, "pred": "SUPPORTED", "format_ok": True,
                     "token_conf": 0.9, "evidence": ["A:1"], "gold": "SUPPORTED", "correct": True},
                    {"draw_index": 2, "pred": "REFUTED", "format_ok": True,
                     "token_conf": 0.9, "evidence": [], "gold": "SUPPORTED", "correct": False},
                ]},
                {"evidence": [{"id": "B:2", "text": "second"}], "draws": [
                    {"draw_index": i, "pred": "SUPPORTED", "format_ok": True,
                     "token_conf": 0.9, "evidence": ["B:2"]} for i in range(3)
                ]},
            ],
        }

    def test_heuristic_retries_same_chunk_then_keeps_both_chunks(self):
        result = select(self.row())
        self.assertEqual(result["extra_draws"], 1)
        self.assertEqual(result["chunk_verifier_calls"], 3)
        self.assertEqual([x["id"] for x in result["selected_evidence"]], ["A:1", "B:2"])
        self.assertEqual([x["selected_draw_index"] for x in result["chunks"]], [1, 0])

    def test_jev_state_hides_gold_and_future_draws(self):
        row = self.row()
        state = json.loads(state_for(row, 0, 0))
        self.assertEqual(len(state["observed_verifier_outputs"]), 1)
        self.assertNotIn("current_group_sentences", state)
        self.assertNotIn("raw_output", state)
        self.assertNotIn("second", json.dumps(state))
        self.assertNotIn("gold", state)
        self.assertNotIn("correct", state)
        controller = FakeController()
        result = select(row, controller)
        self.assertEqual(result["controller_calls"], 3)
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(controller.states[0]["remaining_retries_for_this_group"], 2)

    def test_third_draw_forces_stop(self):
        row = self.row()
        row["chunks"] = row["chunks"][:1]
        for draw in row["chunks"][0]["draws"]:
            draw["pred"] = "NOT_ENOUGH_INFO"
            draw["evidence"] = []
        result = select(row)
        self.assertEqual(result["forced_stops_after_retry_limit"], 1)
        self.assertEqual(result["chunk_verifier_calls"], 3)
        self.assertEqual(result["selected_evidence"], [])

    def test_confidence_is_not_semantic_quality(self):
        draw = {"format_ok": True, "pred": "SUPPORTED", "token_conf": 0.9,
                "evidence": ["A:1"]}
        self.assertEqual(heuristic_action(draw), STOP)

    def test_full_evaluation_writes_separate_threshold_runs(self):
        row = self.row()
        row["chunks"] = row["chunks"][:1]
        row["shown"] = row["shown"][:1]
        row["gold_in_context"] = True
        class FakeVerifier:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, prompts, seed):
                return [{"prompt": prompts[0]}]

        def final_score(claim, evidence, generation):
            ok = bool(evidence)
            return {"gold": "SUPPORTED", "pred": "SUPPORTED" if ok else "NOT_ENOUGH_INFO",
                    "correct": ok, "fever_score": ok, "evidence_f1": float(ok),
                    "format_ok": True}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            manifest = {"evaluation_code": "agent.data.build_evidence_batches",
                        "batches_only": True, "draws_per_chunk": 3, "do_sample": True,
                        "batch_size": 10, "n_claims": 1, "claim_seed": 0,
                        "generation_seed": 0, "model": "fake", "adapter": "fake",
                        "prompt_style": "direct", "max_new_tokens": 160,
                        "model_batch_size": 4}
            (source / "manifest.json").write_text(json.dumps(manifest))
            (source / "metrics.json").write_text('{"n_claims":1}')
            (source / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            output = root / "results" / "rsi-jev-retry-baseline" / "runs" / "fake-model" / "direct-cap160" / "run1"
            fake_torch = types.SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.object(evaluator, "sample_claims", return_value=[{"id": 1, "claim": "A claim"}]), \
                    patch.object(evaluator, "load_controller", return_value=FakeController()), \
                    patch.object(evaluator, "LocalModel", FakeVerifier), \
                    patch.object(evaluator, "scored", side_effect=final_score), \
                    patch.object(evaluator, "gold_sets", return_value=[{("A", 1)}]), \
                    patch.object(evaluator.importlib.metadata, "version", return_value="test"), \
                    patch.dict(sys.modules, {"torch": fake_torch}):
                evaluator.run(SimpleNamespace(source_run=str(source), output_dir=str(output),
                                              heuristic_results_root=str(root / "results"),
                                              backend="rsi-jev", model="fake-jev"))
            self.assertTrue((output / "metrics.json").exists())
            for threshold in ("0p3", "0p5", "0p7"):
                folder = root / "results" / f"heuristic-retry-threshold-{threshold}" / "runs" / "fake-model" / "direct-cap160" / "run1"
                self.assertTrue((folder / "metrics.json").exists())
                self.assertEqual(json.loads((folder / "metrics.json").read_text())["quality"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
