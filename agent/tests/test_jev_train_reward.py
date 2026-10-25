import unittest
from unittest.mock import patch

from agent.train.train_jev_rl import episode_reward
from agent.train.replay_env import RewardWeights


class JevTrainRewardTest(unittest.TestCase):
    def test_final_fever_and_extra_draw_cost(self):
        claim = {"id": 1, "label": "SUPPORTS"}
        final = {"fever_score": True, "correct": True}
        with patch("agent.train.train_jev_rl.gold_sets", return_value=[[('A', 1)]]):
            reward, penalty = episode_reward(
                final, ["A:1"], ["A:1", "B:2"], claim, 2, RewardWeights())
        self.assertEqual(penalty["missing"], 0)
        self.assertAlmostEqual(reward, 0.9)

    def test_missing_reachable_gold_reduces_reward(self):
        claim = {"id": 1, "label": "SUPPORTS"}
        final = {"fever_score": False, "correct": True}
        with patch("agent.train.train_jev_rl.gold_sets", return_value=[[('A', 1)]]):
            reward, penalty = episode_reward(
                final, [], ["A:1"], claim, 0, RewardWeights())
        self.assertGreater(penalty["penalty"], 0)
        self.assertLess(reward, 0)


if __name__ == "__main__":
    unittest.main()
