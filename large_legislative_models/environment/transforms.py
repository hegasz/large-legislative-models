import torch
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class TaxTransform(BaseParallelWrapper):
    """ Wrapper for adding taxation to meltingpot environments. """
    def __init__(self, env):
        super().__init__(env)
        self.total_agents = env.max_num_agents
        self.reward_init()

    def reward_init(self):
        self.reward_history = {f"player_{num}": [] for num in range(self.total_agents)}

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.reward_init()
        # returns tuple of obs and info
        return self.env.reset(seed=seed, options=options)

    def tax_function(self, untaxed, window_length=10):
        """
        We use rewards summed window_length to determine the period over which to consider
        the wealth of agents, analagous to being taxed different amounts based
        on different rates of income, rather than money in a bank.
        """

        # untaxed_reward is the raw reward from harvest_commons, we unpack into a tensor
        untaxed_reward = torch.tensor([untaxed[agent].item() for agent in untaxed], dtype=torch.float32)
        tax_brackets = [0.0, 0.3 * window_length, 0.5 * window_length, float("inf")]
        # sum rewards in last 'window' steps for each agent
        window_reward = torch.tensor(
            [
                sum(self.reward_history[agent][-window_length:]) + untaxed_reward[i]
                for i, agent in enumerate(self.reward_history)
            ],
            dtype=torch.float32,
        )
        
        tax = torch.zeros_like(untaxed_reward)
        for i in range(len(tax_brackets) - 1):
            mask = (window_reward >= tax_brackets[i]) & (window_reward < tax_brackets[i + 1])
            # note this only taxes the reward at this step
            tax[mask] += self.tax_rate[i] * untaxed_reward[mask]

        """ Scale taxes. """
        tax *= 4

        """ Apply and evenly redistribute taxes. """
        taxed_reward = untaxed_reward - tax
        taxed_reward += torch.ones_like(window_reward) * tax.mean()

        taxed_reward_dict = {agent: taxed_reward[i].item() for i, agent in enumerate(self.reward_history)}
        return taxed_reward_dict, taxed_reward, untaxed_reward

    def step(self, action):
        obs, rew, termination, truncation, info = super().step(action)
        current_num_apples = float(obs["player_0"]["LIVE_APPLE_COUNT"])

        """ All infos passed up through player 0. Other players' infos are discarded. """
        info["player_0"]["current_num_apples"] = current_num_apples

        taxed_dict, taxed_tensor, raw_reward_tensor = self.tax_function(rew)

        info["player_0"]["raw_rewards"] = raw_reward_tensor

        """ Keep history of raw rewards for tax calculation. """
        for player, raw_reward in rew.items():
            self.reward_history[player].append(raw_reward)

        for player, reward in taxed_dict.items():
            rew[player] = reward

        taxed_reward_singletons = {agent: taxed_tensor[i] for i, agent in enumerate(obs.keys())}
        return obs, taxed_reward_singletons, termination, truncation, info

    def apply_principal_action(self, tax_rates):
        if any(i < 0 for i in tax_rates) or any(i > 1 for i in tax_rates):
            raise ValueError("Tax rates must be between 0 and 1")
        self.tax_rate = tax_rates


class IncentiveTransform(BaseParallelWrapper):
    """ Wrapper for adding action incentivisation to meltinpot environments. """
    def __init__(self, env, initial_num_apples):
        super().__init__(env)
        self.total_agents = env.max_num_agents
        self.initial_num_apples = initial_num_apples
        self.prev_num_apples = initial_num_apples

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.prev_num_apples = self.initial_num_apples
        return self.env.reset(seed=seed, options=options)

    def apply_incentive(self, raw_reward, obs):
        """
        We want to sort agents into three categories at each step: Those who
        have just collected an apple, those who have just cleaned pollution,
        and those who have done neither. These categories correspond to the self.incentives,
        which is a tensor of shape (3).

        raw_reward is a dict of players to their raw rewards at this step.

        obs is a dict of players, where each player is a dict with keys
        'COLLECTIVE_REWARD', 'PLAYER_CLEANED', 'READY_TO_SHOOT', 'RGB', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP'.

        'PLAYER_CLEANED' == 1 if the player fired their cleaning ray AND cleaned pollution (didn't fire at grass/apples/wall)
        """

        raw_reward_tensor = torch.tensor([raw_reward[agent].item() for agent in raw_reward], dtype=torch.float32)
        num_apples_consumed = torch.sum(raw_reward_tensor == 1).item()

        step_incentives = torch.zeros_like(raw_reward_tensor)
        """ Add incentive for agents that consumed an apple this step. """
        step_incentives[raw_reward_tensor == 1] += self.incentives[0]
        raw_reward_tensor[raw_reward_tensor == 1] -= 0.9
        """ For agents that did not eat an apple this step, determine the incentive
        based on whether they cleaned pollution or took some other action. """
        num_cleaned = 0
        for idx, agent in enumerate(raw_reward):
            if raw_reward_tensor[idx] == 0:
                if obs[agent]["PLAYER_CLEANED"] == 1:
                    num_cleaned += 1
                    raw_reward_tensor[idx] -= 1.0
                    step_incentives[idx] += self.incentives[1]
                else:
                    step_incentives[idx] += self.incentives[2]

        incentivized_tensor = raw_reward_tensor + step_incentives

        """ Returns incentivised rewards, total incentive given, number of agents that cleaned, raw reward. """
        return incentivized_tensor, step_incentives.sum(), num_cleaned, raw_reward_tensor, num_apples_consumed

    def step(self, action):
        obs, rew, termination, truncation, info = super().step(action)
        current_num_apples = float(obs["player_0"]["LIVE_APPLE_COUNT"])

        incentivized_tensor, total_incentive_given, num_cleaned, raw_reward_tensor, num_apples_consumed = (
            self.apply_incentive(rew, obs)
        )

        apples_regrown = current_num_apples + num_apples_consumed - self.prev_num_apples
        self.prev_num_apples = current_num_apples

        """ All infos passed up through player 0. Other players' infos are discarded. """
        info["player_0"]["current_num_apples"] = current_num_apples
        info["player_0"]["total_incentive_given"] = total_incentive_given
        info["player_0"]["num_cleaned"] = num_cleaned
        info["player_0"]["raw_rewards"] = raw_reward_tensor
        info["player_0"]["apples_regrown"] = apples_regrown

        incentivized_singletons = {agent: incentivized_tensor[i] for i, agent in enumerate(obs.keys())}
        return obs, incentivized_singletons, termination, truncation, info

    def apply_principal_action(self, incentives):
        """
        incentives is a tensor of shape (3), where:
        incentives[0] is the incentive for agents that consumed an apple this step
        incentives[1] is the incentive for agents that cleaned pollution this step
        incentives[2] is the incentive for agents that did neither of the above
        """
        self.incentives = incentives
