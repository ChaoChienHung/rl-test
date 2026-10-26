"""Checks that controller SFT split and hindsight labels have the intended meaning."""

import json
import unittest
from pathlib import Path

from agent.data.build_controller_split import DEV, OUTPUT, SOURCE, make_manifest
from agent.train.build_citation_sft_data import (
    citation_score, label_from_scores, pooled_action_label,
)


class JevSftPipelineTest(unittest.TestCase):
    def test_committed_train_split_is_balanced_disjoint_and_dev_free(self):
        manifest = json.loads(OUTPUT.read_text())
        self.assertEqual(manifest, make_manifest(SOURCE, DEV))
        self.assertEqual(len(manifest["sft_ids"]), 3000)
        self.assertEqual(len(manifest["rl_ids"]), 2500)
        self.assertFalse(set(manifest["sft_ids"]) & set(manifest["rl_ids"]))
        self.assertEqual(set(manifest["counts"]["sft"].values()), {1000})

    def test_partial_chunk_gold_can_improve_without_being_complete(self):
        groups = [{"A:1", "A:2"}]
        one = {"format_ok": True, "evidence": ["A:1"]}
        both = {"format_ok": True, "evidence": ["A:1", "A:2"]}
        score_one = citation_score(one, "SUPPORTED", groups, {"A:1", "A:2"})
        score_both = citation_score(both, "SUPPORTED", groups, {"A:1", "A:2"})
        self.assertEqual(label_from_scores(score_one, [score_both], 0)[0], "RETRY")
        self.assertIsNone(citation_score(one, "NOT_ENOUGH_INFO", groups, {"A:1", "A:2"}))
        self.assertIsNone(citation_score(one, "SUPPORTED", groups, {"B:1"}))

    def test_pooled_reward_labels_use_actual_recovery_and_omit_futile_retry(self):
        wrong = {"gold": "SUPPORTED", "correct": False, "fever_score": False}
        right = {"gold": "SUPPORTED", "correct": True, "fever_score": True}
        self.assertEqual(pooled_action_label([wrong, right, wrong], [-1, 1, -1], 0, 0.2)[0], "RETRY")
        self.assertEqual(pooled_action_label([right, wrong, wrong], [1, -1, -1], 0, 0.2)[0], "STOP")
        self.assertIsNone(pooled_action_label([wrong, wrong, wrong], [-1, -1, -1], 0, 0.2))


if __name__ == "__main__":
    unittest.main()
