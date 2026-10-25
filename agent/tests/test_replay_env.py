import unittest

from agent.train.replay_env import ClaimReplayEpisode, RETRY, STOP, ReplayEpisode


class ReplayEnvTest(unittest.TestCase):
    context = {"claim": "A claim", "evidence": ["sentence 1"]}

    def test_hidden_future_and_budget(self):
        draws = [{"verdict": "NEI", "draw": i} for i in range(3)]
        finals = [{"fever_score": False, "correct": False},
                  {"fever_score": True, "correct": True},
                  {"fever_score": True, "correct": True}]
        episode = ReplayEpisode(self.context, draws, finals)
        self.assertEqual(episode.state()["observed_draws"], draws[:1])
        self.assertEqual(episode.state()["evidence"], ["sentence 1"])
        self.assertNotIn("final", episode.state())
        self.assertEqual(episode.step(RETRY)["state"]["observed_draws"], draws[:2])
        self.assertEqual(episode.step(RETRY)["state"]["actions"], [STOP])
        with self.assertRaises(ValueError):
            episode.step(RETRY)
        result = episode.step(STOP)
        self.assertAlmostEqual(result["reward"], 0.9)
        self.assertEqual(result["draws_used"], 3)
        with self.assertRaises(RuntimeError):
            episode.step(STOP)

    def test_stop_reward_uses_final_verification(self):
        draws = [{"pred": "SUPPORTED"}] * 3
        finals = [{"fever_score": False, "correct": True}] * 3
        episode = ReplayEpisode(self.context, draws, finals)
        self.assertEqual(episode.step(STOP)["reward"], 0.0)

    def test_claim_episode_reads_every_chunk_and_hides_gold_from_policy(self):
        chunks = [
            {"evidence": [{"id": "Page:1", "text": "one"}],
             "draws": [{"pred": "NOT_ENOUGH_INFO", "evidence": [], "gold": "SUPPORTED", "correct": False},
                       {"pred": "SUPPORTED", "evidence": ["Page:1"], "gold": "SUPPORTED", "correct": True},
                       {"pred": "REFUTED", "evidence": [], "gold": "SUPPORTED", "correct": False}]},
            {"evidence": [{"id": "Page:2", "text": "two"}],
             "draws": [{"pred": "SUPPORTED", "evidence": ["Page:2"], "gold": "SUPPORTED", "correct": True}] * 3},
        ]
        saved_at_final = []

        def finalize(saved):
            saved_at_final.extend(saved)
            return {"fever_score": True, "correct": True}

        episode = ClaimReplayEpisode("A claim", chunks, [{"Page:1"}], finalize)
        self.assertNotIn("gold", episode.state()["observed_draws"][0])
        self.assertNotIn("correct", episode.state()["observed_draws"][0])
        self.assertEqual(episode.step(RETRY)["state"]["chunk_index"], 0)
        self.assertEqual(episode.step(STOP)["state"]["chunk_index"], 1)
        result = episode.step(STOP)
        self.assertEqual(saved_at_final, ["Page:1", "Page:2"])
        self.assertTrue(result["done"])
        self.assertEqual(result["extra_draws"], 1)


if __name__ == "__main__":
    unittest.main()
