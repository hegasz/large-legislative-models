import numpy as np
import torch
from large_legislative_models.principal import Principal


class BanditWrapper(Principal):
    """ Wrapper for bandit algorithms acting as principals. """

    def __init__(self, args, bandit, num_brackets):
        super().__init__(args)
        if args.num_parallel_games > 1:
            raise NotImplementedError

        """ Discretized output range for each bracket. """
        self.discretization_set = args.discretization_set

        """  Action space upper bound for each bracket. """
        self.upper_bound = args.upper_bound

        """ Bandit algorithm wrapped. """
        self.bandit = bandit

        """ Number of tax brackets / incentives in incentive set. """
        self.num_brackets = num_brackets

    def set_tax_vals(self, indicator, principal_step):
        """Decide and set tax values / incentives.

        Args:
            indicator (Tensor): one-hot encoding of environment indicator
            principal_step (int): current principal step number

        Returns:
            (Tensor): tax rates / incentives chosen
        """

        """ Convert one-hot indicator to integer and save in field for update step. """
        if indicator.shape[-1] == 0:
            self.indicator_int = 0
        else:
            self.indicator_int = indicator.argmax(dim=-1).item()

        """ Query bandit algorithm for an action. """
        self.bandit_action = self.bandit.get_action(self.indicator_int)

        """ Convert action number into corresponding set of tax rates. """
        tax_vals_per_game = self.convert_action(self.bandit_action)

        return tax_vals_per_game

    def update(self, ctx, principal_step, episode_buffers):
        """Update bandit parameters using validation episodes.

        Args:
            ctx (Context): training context
            principal_step (int): principal step number
            episode_buffers (list[ValidationEpisode]): list of validation episode buffers
        """
        # shape (num_val_episodes, num_parallel_games) - for measurements across validation episodes, use mean dim=0
        mean_principal_reward_per_game = torch.stack([ep.principal_cumulative_reward for ep in episode_buffers]).mean(
            dim=0
        )
        self.bandit.update_params(self.indicator_int, self.bandit_action, mean_principal_reward_per_game)

    def convert_action(self, action):
        """ Converts a numbered bandit arm into corresponding set of tax rates / incentives.
        Set of all discretized tax rates / incentives is ordered numerically and bandit arm
        is each set's corresponding index in this order.
        e.g. 0 -> [0,0,0]; 1 -> [0,0,0.1]; ... 

        Args:
            action (int): number of bandit arm chosen

        Returns:
            (Tensor): tax rates / incentive set
        """
        rates = []

        """ Base of the "counting" system. """
        base = len(self.discretization_set)

        for _ in range(self.num_brackets):
            index = action % base  # index for the current rate
            rates.append(self.discretization_set[index])  # map index to the corresponding rate value
            action //= base  # move to the next place (just like in normal counting)

        return torch.Tensor([rates[::-1]])  # Reverse to get the correct order

    def save_params(self, principal_step):
        self.bandit.save_params(principal_step)


class EpsilonGreedy:
    """ A simple epsilon-greedy bandit algorithm. """

    def __init__(self, num_indicators, arm_count, epsilon, seed, stepsize="avg"):
        self.epsilon = epsilon
        self.seed = seed
        self.arm_count = arm_count
        
        """ One row for each indicator value, converting contextual bandits to vanilla bandits. """
        self.Q = np.zeros((num_indicators, arm_count))
        self.N = np.zeros((num_indicators, arm_count))
        if stepsize == "avg":
            self.get_stepsize = lambda indicator, arm: 1 / self.N[indicator, arm]
        else:
            self.get_stepsize = lambda indicator, arm: stepsize

    def get_action(self, indicator):
        if np.random.uniform(0, 1) > self.epsilon:
            action = self.Q[indicator].argmax()
        else:
            action = np.random.randint(0, self.arm_count)
        return action

    def update_params(self, indicator, arm, reward):
        self.N[indicator, arm] += 1
        self.Q[indicator, arm] += self.get_stepsize(indicator, arm) * (reward.item() - self.Q[indicator, arm])

    def save_params(self, principal_step):
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_eps_greedy_Q.npy", self.Q)
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_eps_greedy_N.npy", self.N)


class UCB:
    """ Upper confidence bound bandit algorithm. """

    def __init__(self, num_indicators, arm_count, coef, seed, stepsize="avg"):
        self.coef = coef
        self.seed = seed
        self.arm_count = arm_count
        
        """ One row for each indicator value, converting contextual bandits to vanilla bandits. """
        self.Q = np.zeros((num_indicators, arm_count))
        self.N = np.zeros((num_indicators, arm_count))
        if stepsize == "avg":
            self.get_stepsize = lambda indicator, arm: 1 / self.N[indicator, arm]
        else:
            self.get_stepsize = lambda indicator, arm: stepsize
        self.timestep = 0

    def get_action(self, indicator):
        """ Increment timestep. """
        self.timestep += 1

        """ All arms need to be played once to begin with.
        Can make this more concise by adding a small epsilon to self.N
        at initialisation, but this is clearer. """
        for arm in range(self.arm_count):
            if self.N[indicator, arm] == 0:
                return arm

        """ Calculate confidence bounds. """
        ln_timestep = np.log(np.full(self.arm_count, self.timestep))
        confidence = self.coef * np.sqrt(ln_timestep / self.N[indicator])

        """ Choose argmax of value plus confidence bound. """
        action = np.argmax(self.Q[indicator] + confidence)
        return action

    def update_params(self, indicator, arm, reward):
        self.N[indicator, arm] += 1
        self.Q[indicator, arm] += self.get_stepsize(indicator, arm) * (reward - self.Q[indicator, arm])

    def save_params(self, principal_step):
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_ucb_Q.npy", self.Q)
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_ucb_N.npy", self.N)


class ThompsonSampling:
    """Arm distributions assumed Gaussian with unknown mean and variance.
    Conjugate prior is normal-inverse-gamma NIG(mean, count, shape, scale)
    See: https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution
    Posterior derivation from excellent article https://gertjanvandenburg.com/blog/thompson_sampling/"""

    def __init__(self, num_indicators, arm_count, seed, prior_mean=0, prior_count=0.05, prior_shape=1, prior_scale=25):
        self.seed = seed
        self.prior_mean = prior_mean
        self.prior_count = prior_count
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale

        """ One row for each indicator value, converting contextual bandits to vanilla bandits. """
        self.N = np.zeros((num_indicators, arm_count))
        self.mean = np.zeros((num_indicators, arm_count))
        self.rho = np.full((num_indicators, arm_count), self.prior_mean)
        self.ssd = np.zeros((num_indicators, arm_count))
        self.beta = np.full((num_indicators, arm_count), self.prior_scale)

    def get_action(self, indicator):
        sigma2 = 1.0 / np.random.gamma(0.5 * self.N[indicator] + self.prior_shape, 1.0 / self.beta[indicator])
        mus = np.random.normal(self.rho[indicator], np.sqrt(sigma2 / (self.N[indicator] + self.prior_count)))
        return mus.argmax()

    def update_params(self, indicator, arm, reward):
        old_N, old_mean = self.N[indicator, arm], self.mean[indicator, arm]
        self.N[indicator, arm] += 1
        self.mean[indicator, arm] += 1 / self.N[indicator, arm] * (reward - self.mean[indicator, arm])
        self.rho[indicator, arm] = (
            self.prior_count * self.prior_mean + self.N[indicator, arm] * self.mean[indicator, arm]
        ) / (self.prior_count + self.N[indicator, arm])
        self.ssd[indicator, arm] += (
            reward**2 + old_N * old_mean**2 - self.N[indicator, arm] * self.mean[indicator, arm] ** 2
        )
        self.beta[indicator, arm] = (
            self.prior_scale
            + 0.5 * self.ssd[indicator, arm]
            + (
                self.N[indicator, arm]
                * self.prior_count
                * (self.mean[indicator, arm] - self.prior_mean) ** 2
                / (2 * (self.N[indicator, arm] + self.prior_count))
            )
        )

    def save_params(self, principal_step):
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_thompson_N.npy", self.N)
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_thompson_mean.npy", self.mean)
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_thompson_rho.npy", self.rho)
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_thompson_ssd.npy", self.ssd)
        np.save(f"./saved_params/step{principal_step}/seed_{self.seed}_thompson_beta.npy", self.beta)
