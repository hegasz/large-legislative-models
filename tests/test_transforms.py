import unittest
import torch
from unittest.mock import MagicMock

from large_legislative_models.environment.transforms import IncentiveTransform, TaxTransform


class TestTaxTransform(unittest.TestCase):
    def setUp(self):
        # Create a mock environment
        self.mock_env = MagicMock()
        self.mock_env.max_num_agents = 2
        self.mock_env.reset.return_value = ("obs", "info")
        self.mock_env.step.return_value = ("obs", "rew", "termination", "truncation", "info")
        self.mock_env.agents = [f"player_{i}" for i in range(self.mock_env.max_num_agents)]
        self.mock_env.possible_agents = self.mock_env.agents.copy()

        # Initialize TaxTransform with the mock environment
        self.tax_transform = TaxTransform(self.mock_env)
        # Set known tax rates and brackets for testing
        self.tax_transform.tax_rate = [0.0, 0.1, 0.2]

    def test_initialization(self):
        """Test that the reward history is initialized correctly."""
        expected_reward_history = {f"player_{i}": [] for i in range(self.mock_env.max_num_agents)}
        self.assertEqual(self.tax_transform.reward_history, expected_reward_history)

    def test_reward_init(self):
        """Test that reward_init properly initializes reward history."""
        self.tax_transform.reward_history = {"player_0": [1], "player_1": [2]}
        self.tax_transform.reward_init()
        expected_reward_history = {f"player_{i}": [] for i in range(self.mock_env.max_num_agents)}
        self.assertEqual(self.tax_transform.reward_history, expected_reward_history)

    def test_reset(self):
        """Test that reset reinitializes reward history and calls env.reset."""
        self.tax_transform.reset()
        expected_reward_history = {f"player_{i}": [] for i in range(self.mock_env.max_num_agents)}
        self.assertEqual(self.tax_transform.reward_history, expected_reward_history)
        self.mock_env.reset.assert_called_once()

    def test_apply_principal_action_valid(self):
        """Test that valid tax rates are accepted."""
        valid_tax_rates = [0.1, 0.2, 0.3]
        self.tax_transform.apply_principal_action(valid_tax_rates)
        self.assertEqual(self.tax_transform.tax_rate, valid_tax_rates)

    def test_apply_principal_action_invalid(self):
        """Test that invalid tax rates raise a ValueError."""
        invalid_tax_rates = [-0.1, 1.1, 0.2]
        with self.assertRaises(ValueError):
            self.tax_transform.apply_principal_action(invalid_tax_rates)

    def test_tax_function(self):
        """Test tax_function with controlled inputs to verify taxation logic."""
        # Set up reward history
        self.tax_transform.reward_history = {"player_0": [1.0] * 10, "player_1": [0.0] * 10}
        untaxed = {"player_0": torch.tensor(1.0), "player_1": torch.tensor(1.0)}
        # Run tax_function
        taxed_reward_dict, taxed_reward_tensor, untaxed_reward = self.tax_transform.tax_function(
            untaxed=untaxed, window_length=10
        )
        # player_0 taxed at 0.2 * 4 = 0.8
        # player_1 not taxed
        # total 0.8 collected from just player_0
        # redistribute 0.4 each
        expected_taxed_rewards = {"player_0": 0.6, "player_1": 1.4}
        # Verify the taxed rewards
        for agent in ["player_0", "player_1"]:
            self.assertAlmostEqual(taxed_reward_dict[agent], expected_taxed_rewards[agent], places=5)

    def test_step(self):
        """Test the step function to ensure it applies taxation correctly."""
        # Mock the environment's step output
        obs = {f"player_{i}": {"LIVE_APPLE_COUNT": 10} for i in range(self.mock_env.max_num_agents)}
        rew = {f"player_{i}": torch.tensor(1.0) for i in range(self.mock_env.max_num_agents)}
        termination = {f"player_{i}": False for i in range(self.mock_env.max_num_agents)}
        truncation = {f"player_{i}": False for i in range(self.mock_env.max_num_agents)}
        info = {f"player_{i}": {} for i in range(self.mock_env.max_num_agents)}
        self.tax_transform.env.step.return_value = (obs, rew, termination, truncation, info)
        # Set tax rates
        self.tax_transform.apply_principal_action([0.0, 0.1, 0.2])
        # Call step
        action = {f"player_{i}": None for i in range(self.mock_env.max_num_agents)}
        obs, taxed_reward_singletons, termination, truncation, info = self.tax_transform.step(action)
        # Verify the taxed rewards
        self.assertEqual(len(taxed_reward_singletons), self.mock_env.max_num_agents)
        # Verify that info is updated
        self.assertIn("current_num_apples", info["player_0"])
        self.assertIn("raw_rewards", info["player_0"])


class TestIncentiveTransform(unittest.TestCase):
    def setUp(self):
        # Create a mock environment
        self.mock_env = MagicMock()
        self.mock_env.max_num_agents = 2
        self.mock_env.reset.return_value = ("obs", "info")
        self.mock_env.step.return_value = ("obs", "rew", "termination", "truncation", "info")
        self.mock_env.agents = [f"player_{i}" for i in range(self.mock_env.max_num_agents)]
        self.mock_env.possible_agents = self.mock_env.agents.copy()
        # Initialize IncentiveTransform
        self.incentive_transform = IncentiveTransform(self.mock_env, initial_num_apples=20)
        # Set known incentives for testing
        self.incentive_transform.incentives = torch.tensor([0.5, 0.3, 0.1])

    def test_initialization(self):
        """Test that the initial number of apples is set correctly."""
        self.assertEqual(self.incentive_transform.prev_num_apples, 20)

    def test_reset(self):
        """Test that reset sets prev_num_apples correctly and calls env.reset."""
        self.incentive_transform.reset()
        self.assertEqual(self.incentive_transform.prev_num_apples, 20)
        self.mock_env.reset.assert_called_once()

    def test_apply_principal_action(self):
        """Test that incentives are updated correctly."""
        incentives = torch.tensor([0.1, 0.3, 1.0])
        self.incentive_transform.apply_principal_action(incentives)
        self.assertTrue(torch.equal(self.incentive_transform.incentives, incentives))

    def test_apply_incentive(self):
        """Test apply_incentive with controlled inputs to verify incentive logic."""
        # Mock raw rewards and observations
        raw_reward = {f"player_0": torch.tensor(1.0), f"player_1": torch.tensor(0.0)}
        obs = {f"player_0": {"PLAYER_CLEANED": 0}, f"player_1": {"PLAYER_CLEANED": 0}}
        # Apply incentives
        incentivized_tensor, total_incentive_given, num_cleaned, raw_reward_tensor, num_apples_consumed = (
            self.incentive_transform.apply_incentive(raw_reward, obs)
        )
        # player_0 harvested so gets 0.5 incentive, plus 0.1 base reward
        # player_1 did a "other" action so gets 0.1 incentive
        expected_incentivized_rewards = torch.tensor([0.6, 0.1])
        self.assertTrue(torch.allclose(incentivized_tensor, expected_incentivized_rewards))
        # gave 0.5 + 0.1 = 0.6
        self.assertAlmostEqual(total_incentive_given.item(), 0.6)
        self.assertEqual(num_cleaned, 0)
        self.assertEqual(num_apples_consumed, 1)

    def test_step(self):
        """Test the step function to ensure incentives are applied correctly."""
        # Mock the environment's step output
        obs = {
            f"player_{i}": {"LIVE_APPLE_COUNT": 10, "PLAYER_CLEANED": 0} for i in range(self.mock_env.max_num_agents)
        }
        rew = {f"player_{i}": torch.tensor(1.0) for i in range(self.mock_env.max_num_agents)}
        termination = {f"player_{i}": False for i in range(self.mock_env.max_num_agents)}
        truncation = {f"player_{i}": False for i in range(self.mock_env.max_num_agents)}
        info = {f"player_{i}": {} for i in range(self.mock_env.max_num_agents)}
        self.incentive_transform.env.step.return_value = (obs, rew, termination, truncation, info)
        # Set incentives
        self.incentive_transform.apply_principal_action(torch.tensor([0.5, 0.3, -0.1]))
        # Call step
        action = {f"player_{i}": None for i in range(self.mock_env.max_num_agents)}
        obs, incentivized_singletons, termination, truncation, info = self.incentive_transform.step(action)
        # Verify the incentivized rewards
        self.assertEqual(len(incentivized_singletons), self.mock_env.max_num_agents)
        # Verify that info is updated
        self.assertIn("current_num_apples", info["player_0"])
        self.assertIn("total_incentive_given", info["player_0"])
        self.assertIn("num_cleaned", info["player_0"])
        self.assertIn("raw_rewards", info["player_0"])
        self.assertIn("apples_regrown", info["player_0"])

    def test_apply_incentive_with_cleaners(self):
        """Test apply_incentive when some agents clean pollution."""
        # Mock raw rewards and observations
        raw_reward = {"player_0": torch.tensor(0.0), "player_1": torch.tensor(1.0)}
        obs = {"player_0": {"PLAYER_CLEANED": 1}, "player_1": {"PLAYER_CLEANED": 0}}
        # Apply incentives
        incentivized_tensor, total_incentive_given, num_cleaned, raw_reward_tensor, num_apples_consumed = (
            self.incentive_transform.apply_incentive(raw_reward, obs)
        )
        # Expected calculations
        # player_0 cleaned so gets 0.3 - 1.0 = -0.7
        # player_1 harvested so gets 0.5 + 0.1 = 0.6
        expected_incentivized_rewards = torch.tensor([-0.7, 0.6])
        self.assertTrue(torch.allclose(incentivized_tensor, expected_incentivized_rewards, atol=1e-5))
        # gave 0.3 + 0.5 = 0.8
        self.assertAlmostEqual(total_incentive_given.item(), 0.8, places=5)
        self.assertEqual(num_cleaned, 1)
        self.assertEqual(num_apples_consumed, 1)


if __name__ == "__main__":
    unittest.main()
