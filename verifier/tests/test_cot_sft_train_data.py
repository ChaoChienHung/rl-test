"""Preflight checks for the separate CoT SFT training dataset."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from utils.experiment_layout import ROOT

train_path = ROOT / "verifier/experiments/cot_sft_qwen_3.5-2b_oracle_pilot300/train.py"
spec = importlib.util.spec_from_file_location("cot_sft_train", train_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
load_cot_data = module.load_cot_data


class CotSftTrainDataTests(unittest.TestCase):
    def test_complete_accepted_dataset_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = directory / "cot_sft_labeled_prompts.jsonl"
            row = {"id": "synthetic-train-only", "gold": "SUPPORTED", "prompt": "Claim: A. E1: A.",
                   "completion": json.dumps({"reasoning": "E1 supports the claim.",
                                              "verdict": "SUPPORTED", "evidence": ["E1"]})}
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (directory / "manifest.json").write_text('{"count": 1, "provider": "soclaas", "model": "qwen3.8:27b"}')
            (directory / "summary.json").write_text('{"received": 1, "accepted": 1}')
            rows, _, _, digest = load_cot_data(data)
            self.assertEqual(rows, [row])
            self.assertEqual(len(digest), 64)
            row["completion"] = json.dumps({**json.loads(row["completion"]), "confidence": 1.0})
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid CoT SFT completion"):
                load_cot_data(data)


if __name__ == "__main__":
    unittest.main()
