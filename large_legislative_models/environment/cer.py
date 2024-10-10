import random
import torch

class CER:
    """ Contextual esccpae room env.
    Lever states: 0, 1, 2, ..., num_levers-1
    Door state: num_levers, or -2 (second last state)
    Start state: num_levers+1, or -1 (last state)
    """

    def __init__(self, num_agents, min_at_lever, num_levers, max_steps, fixed_indicator=None):

        self.min_at_lever = min_at_lever
        self.max_steps = max_steps
        self.num_levers = num_levers
        self.num_states = num_levers+2
        self.fixed_indicator = fixed_indicator
        # agents observe everyone's position and one-hot lever indicator
        self.agent_observation_shape = (num_agents*self.num_states + num_levers,)
        self.principal_action_length = self.num_states
        self.num_actions = self.num_states
        self.num_agents = num_agents
        self.action_space_shape = ()
        self.name = "cer"

        """ Inialise "current" state to same as reset state for "have agents moved" check. """
        self.current_state = torch.ones(self.num_agents)

    def update_door_status(self):
        num_at_lever = sum(self.current_state == self.indicator).item()
        self.door_open = num_at_lever >= self.min_at_lever

    def calc_raw_rewards(self):
        didnt_move_mask = self.current_state == self.previous_state
        """ Base reward if door stays shut is a -1 penalty for everyone that moved. """
        rewards_per_action = -torch.ones(self.num_actions)
        if self.door_open:
            """ Anyone at door state gets +10. """
            rewards_per_action[self.num_levers] = 10

        rewards = torch.index_select(rewards_per_action, 0, self.current_state)

        no_move_reward = 0
        if not self.door_open:
            rewards[didnt_move_mask] = no_move_reward

        return rewards

    def calc_agent_rewards(self):
        raw_rewards = self.calc_raw_rewards()
        player_incentives = torch.index_select(self.incentives, 0, self.current_state)
        rewards = raw_rewards + player_incentives
        return rewards, player_incentives.sum()

    def calc_principal_reward(self):
        raw_rewards = self.calc_raw_rewards()
        principal_reward = raw_rewards.sum()
        return principal_reward

    def get_agent_observations(self):
        everybody_position = torch.nn.functional.one_hot(self.current_state.long(), num_classes=self.num_states).flatten()
        lever_one_hot = torch.nn.functional.one_hot(torch.tensor(self.indicator).long(), num_classes=self.num_levers).flatten()
        obs = torch.cat([everybody_position, lever_one_hot]).repeat((self.num_agents, 1))
        return obs.float()

    def get_world_obs(self):
        everybody_position = torch.nn.functional.one_hot(self.current_state.long(), num_classes=self.num_states).flatten()
        return everybody_position

    def determine_whether_done(self):
        return self.step_count == self.max_steps

    def update_state(self, actions):
        self.previous_state = self.current_state
        self.current_state = actions
        self.update_door_status()

    def step(self, actions):
        self.step_count += 1
        self.update_state(actions)
        agent_rewards, total_incentive_given = self.calc_agent_rewards()
        next_agent_obs = self.get_agent_observations()
        done = torch.full((self.num_agents,), self.determine_whether_done())
        info = {
            "world_obs": self.get_world_obs(),
            "indicator": self.indicator,
            "num_start": sum(self.current_state==self.num_levers+1).item(),
            "num_door": sum(self.current_state==self.num_levers).item(),
            "total_raw_reward": self.calc_principal_reward(),
            "total_incentive_given": total_incentive_given,
            "door_open": 1 if self.door_open else 0,
        }
        for i in range(self.num_levers):
            info[f"num_lever_{i}"] = sum(self.current_state == i).item()
        return next_agent_obs, agent_rewards, done, info

    def set_indicator(self):
        if self.fixed_indicator is None:
            indicator = random.randrange(self.num_levers)
        else:
            indicator = self.fixed_indicator
        self.indicator = indicator
        return indicator

    def reset(self):
        self.update_state(torch.full((self.num_agents,), self.num_levers + 1))
        self.step_count = 0
        next_agent_obs = self.get_agent_observations()
        reset_info = {"world_obs": self.get_world_obs}
        return next_agent_obs, reset_info

    def apply_principal_action(self, incentives):
        self.incentives = incentives


class ParallelCER():
    """ Synchronously run parallel CER environments. """
    def __init__(self, num_parallel_games, num_agents, min_at_lever, num_levers, max_steps, fixed_indicator=None):

        self.envs = [
            CER(num_agents, min_at_lever, num_levers, max_steps, fixed_indicator)
            for _ in range(num_parallel_games)
        ]

        self.agent_observation_shape = self.envs[0].agent_observation_shape
        self.principal_action_length = self.envs[0].principal_action_length
        self.action_space_shape  = self.envs[0].action_space_shape 
        self.num_actions = self.envs[0].num_actions
        self.num_levers = self.envs[0].num_levers
        self.num_agents = self.envs[0].num_agents
        self.name = self.envs[0].name

    def apply_principal_action(self, incentives):
        for i, env in enumerate(self.envs):
            env.apply_principal_action(incentives[i])

    def set_indicator(self):
        raw_indicator_per_game = [env.set_indicator() for env in self.envs]
        one_hot = torch.nn.functional.one_hot(torch.tensor(raw_indicator_per_game).long(), num_classes=self.num_levers).float()
        return one_hot

    def reset(self):
        all_obs, all_infos = zip(*(env.reset() for env in self.envs))
        return torch.concatenate(all_obs), list(all_infos)

    def step(self, actions):
        actions = torch.from_numpy(actions)
        all_obs = []
        all_rewards = []
        all_done = []
        all_infos = []

        idx = 0
        for env in self.envs:
            obs, rewards, done, info = env.step(actions[idx: idx+env.num_agents])
            all_obs.append(obs)
            all_rewards.append(rewards)
            all_done.append(done)
            all_infos.append(info)
            idx += env.num_agents

        return (
            torch.cat(all_obs),
            torch.cat(all_rewards, dim=0),
            torch.cat(all_done),
            torch.cat(all_done),
            all_infos
        )
