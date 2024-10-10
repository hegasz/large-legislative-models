import unittest
import torch
import random

from large_legislative_models.environment.cer import CER


class TestCER(unittest.TestCase):
    def setUp(self):
        # Set a seed for reproducibility
        random.seed(0)
        torch.manual_seed(0)

        # Initialize the CER environment with known parameters
        self.num_agents = 3
        self.min_at_lever = 2
        self.num_levers = 2
        self.max_steps = 5
        self.fixed_indicator = None  # For testing random indicator

        self.env = CER(
            num_agents=self.num_agents,
            min_at_lever=self.min_at_lever,
            num_levers=self.num_levers,
            max_steps=self.max_steps,
            fixed_indicator=self.fixed_indicator,
        )
        self.env.set_indicator()

        # Initialize incentives (principal's action)
        self.incentives = torch.zeros(self.env.num_states)
        self.env.apply_principal_action(self.incentives)

    def test_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertEqual(self.env.num_agents, self.num_agents)
        self.assertEqual(self.env.min_at_lever, self.min_at_lever)
        self.assertEqual(self.env.num_levers, self.num_levers)
        self.assertEqual(self.env.max_steps, self.max_steps)
        self.assertEqual(self.env.current_state.shape[0], self.num_agents)
        self.assertEqual(self.env.current_state.tolist(), [1.0, 1.0, 1.0])

    def test_set_indicator(self):
        """Test that the indicator is set correctly."""
        self.env.fixed_indicator = None
        self.env.set_indicator()
        self.assertIn(self.env.indicator, range(self.env.num_levers))

        # Test with fixed indicator
        self.env.fixed_indicator = 0
        self.env.set_indicator()
        self.assertEqual(self.env.indicator, 0)

    def test_reset(self):
        """Test that reset initializes the environment state correctly."""
        obs, info = self.env.reset()
        expected_state = torch.full((self.num_agents,), self.num_levers + 1)
        self.assertTrue(torch.equal(self.env.current_state, expected_state))
        self.assertEqual(self.env.step_count, 0)
        self.assertIn("world_obs", info)

    def test_update_state(self):
        """Test that the state updates correctly given actions."""
        actions = torch.tensor([0, 1, 2])
        self.env.update_state(actions)
        self.env.set_indicator()
        self.assertTrue(torch.equal(self.env.current_state, actions))
        self.assertTrue(torch.equal(self.env.previous_state, torch.ones(self.num_agents)))
        self.env.update_door_status()
        self.assertFalse(self.env.door_open)

    def test_update_door_status(self):
        """Test door status updates based on agents at the indicator lever."""
        # Case when door should not open
        self.env.fixed_indicator = 1
        self.env.current_state = torch.tensor([0, 0, 1])  # Only one agent at indicator
        self.env.update_door_status()
        self.assertFalse(self.env.door_open)

        # Case when door should open
        self.env.fixed_indicator = 1
        self.env.set_indicator()
        self.env.current_state = torch.tensor([1, 1, 0])  # Two agents at indicator (indicator=1)
        self.env.update_door_status()
        self.assertTrue(self.env.door_open)

    def test_calc_raw_rewards(self):
        """Test raw reward calculations under different scenarios."""
        # Scenario where door is closed and some agents moved
        self.env.door_open = False
        self.env.current_state = torch.tensor([0, 1, 2])
        self.env.previous_state = torch.tensor([1, 1, 1])
        raw_rewards = self.env.calc_raw_rewards()
        # door is closed, so no-move reward applies to agent that has not moved.
        expected_rewards = torch.tensor([-1.0, 0, -1.0])
        self.assertTrue(torch.equal(raw_rewards, expected_rewards))
        
        # Scenario where door is open and some agents moved
        self.env.door_open = True
        self.env.current_state = torch.tensor([0, 1, 1])
        self.env.previous_state = torch.tensor([1, 1, 1])
        raw_rewards = self.env.calc_raw_rewards()
        # door is open, so no-move reward does not apply.
        expected_rewards = torch.tensor([-1.0, -1.0, -1.0])
        self.assertTrue(torch.equal(raw_rewards, expected_rewards))

        # Scenario where door is open and agents at door
        self.env.door_open = True
        self.env.current_state = torch.tensor([self.env.num_levers, 1, 0])  # One agent at door
        raw_rewards = self.env.calc_raw_rewards()
        expected_rewards = torch.tensor([10.0, -1.0, -1.0])
        self.assertTrue(torch.equal(raw_rewards, expected_rewards))

    def test_calc_agent_rewards(self):
        """Test agent rewards including incentives."""
        # Set incentives
        self.env.incentives = torch.tensor([0.5] * self.env.num_states)

        # Scenario where door is closed
        self.env.door_open = False
        self.env.current_state = torch.tensor([0, 1, 2])
        self.env.previous_state = torch.tensor([0, 1, 2])
        rewards, total_incentive_given = self.env.calc_agent_rewards()
        raw_rewards = self.env.calc_raw_rewards()
        expected_rewards = raw_rewards + self.env.incentives[self.env.current_state.long()]
        self.assertTrue(torch.equal(rewards, expected_rewards))
        self.assertAlmostEqual(total_incentive_given.item(), 1.5)

    def test_calc_principal_reward(self):
        """Test principal's reward calculation."""
        self.env.update_state(torch.tensor([0, 1, self.env.num_levers]))  # One agent at door
        self.env.door_open = True
        principal_reward = self.env.calc_principal_reward()
        self.assertEqual(principal_reward.item(), 8.0)  # -1 + -1 + 10

    def test_get_agent_observations(self):
        """Test that agent observations are generated correctly."""
        obs = self.env.get_agent_observations()
        self.assertEqual(obs.shape, (self.num_agents, self.num_agents * self.env.num_states + self.num_levers))

    def test_get_world_obs(self):
        """Test that the world observation is correct."""
        world_obs = self.env.get_world_obs()
        expected_obs = torch.nn.functional.one_hot(
            self.env.current_state.long(), num_classes=self.env.num_states
        ).flatten()
        self.assertTrue(torch.equal(world_obs, expected_obs))

    def test_determine_whether_done(self):
        """Test that the environment correctly determines if it is done."""
        self.env.step_count = self.env.max_steps
        done = self.env.determine_whether_done()
        self.assertTrue(done)

        self.env.step_count = self.env.max_steps - 1
        done = self.env.determine_whether_done()
        self.assertFalse(done)

    def test_step(self):
        """Test the step function for correct state transitions and rewards."""
        self.env.reset()
        actions = torch.tensor([1, 1, 0])  # Two agents at indicator (indicator=1)
        next_obs, rewards, done, info = self.env.step(actions)

        # Check state update
        self.assertTrue(torch.equal(self.env.current_state, actions))
        self.assertEqual(self.env.step_count, 1)

        # Check door status
        self.assertTrue(self.env.door_open)

        # Check rewards
        expected_rewards = torch.tensor([-1.0, -1.0, -1.0]) + self.env.incentives[actions.long()]
        self.assertTrue(torch.equal(rewards, expected_rewards))

        # Check if done
        expected_done = torch.tensor([False, False, False])
        self.assertTrue(torch.equal(done, expected_done))

        # Check info dictionary
        self.assertIn("world_obs", info)
        self.assertIn("indicator", info)
        self.assertIn("door_open", info)
        self.assertEqual(info["door_open"], 1)
        self.assertEqual(info["num_door"], 0)

    def test_apply_principal_action(self):
        """Test that the principal's incentives are applied correctly."""
        new_incentives = torch.tensor([0.1] * self.env.num_states)
        self.env.apply_principal_action(new_incentives)
        self.assertTrue(torch.equal(self.env.incentives, new_incentives))

    def test_full_episode(self):
        """Test a full episode run through the environment."""
        self.env.reset()
        total_rewards = torch.zeros(self.num_agents)
        for _ in range(self.env.max_steps):
            actions = torch.randint(0, self.env.num_states, (self.num_agents,))
            next_obs, rewards, done, info = self.env.step(actions)
            total_rewards += rewards
            if done[0]:
                break
        self.assertEqual(self.env.step_count, self.max_steps)
        self.assertTrue(done.all())

    def test_agents_didnt_move(self):
        """Test the case when agents don't move and door is closed."""
        self.env.reset()
        self.env.previous_state = torch.tensor([1, 1, 1])
        self.env.current_state = torch.tensor([1, 1, 1])
        self.env.door_open = False
        rewards = self.env.calc_raw_rewards()
        expected_rewards = torch.tensor([0.0, 0.0, 0.0])  # No penalty for not moving
        self.assertTrue(torch.equal(rewards, expected_rewards))

    def test_agents_at_door_but_door_closed(self):
        """Test the case when agents move to door but door is closed."""
        self.env.reset()
        door_state = self.env.num_levers
        self.env.previous_state = torch.tensor([1, 1, 1])
        self.env.current_state = torch.tensor([door_state, door_state, door_state])
        self.env.door_open = False
        rewards = self.env.calc_raw_rewards()
        expected_rewards = torch.tensor([-1.0, -1.0, -1.0])  # Penalty for moving to door when closed
        self.assertTrue(torch.equal(rewards, expected_rewards))

    def test_agents_at_door_and_door_open(self):
        """Test the case when agents move to door and door is open."""
        self.env.reset()
        door_state = self.env.num_levers
        self.env.previous_state = torch.tensor([1, 1, 1])
        self.env.current_state = torch.tensor([door_state, door_state, door_state])
        self.env.door_open = True
        rewards = self.env.calc_raw_rewards()
        expected_rewards = torch.tensor([10.0, 10.0, 10.0])  # Reward for moving to door when open
        self.assertTrue(torch.equal(rewards, expected_rewards))


if __name__ == "__main__":
    unittest.main()
