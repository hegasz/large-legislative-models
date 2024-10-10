import torch
from large_legislative_models.utils.buffer import FixedLengthTrajectory


def gae_advantage_and_return_estimates(trajectory: FixedLengthTrajectory, next_done, next_value, gamma, gae_lambda):
    """Compute GAE advantage and returns estimates. Advantages have gradient flowing through rewards, returns are detached.

    Args:
        trajectory (FixedLengthTrajectory): trajectory to compute estimates for
        next_done (Tensor[num_parallel_games*num_agents]): done flags of final state for each agent
        next_value (Tensor[num_parallel_games*num_agents]): value estimates of final state for each agent
        gamma (float): discount factor
        gae_lambda (float): GAE lambda coefficient

    Returns:
        advantages (Tensor[trajectory_length, num_parallel_games*num_agents]): GAE advantage estimates
        returns (Tensor[trajectory_length, num_parallel_games*num_agents]): GAE returns estimates
    """

    advantages = torch.zeros_like(trajectory.tensordict["rewards"])

    for t in reversed(range(len(advantages))):
        # nextnonterminal is a boolean flag indicating if the episode ends on the next step
        # nextvalues holds the estimated value of the next state
        if t == len(advantages) - 1:  # last time step
            nextnonterminal = 1.0 - next_done.long()
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - trajectory.tensordict["dones"][t + 1]
            nextvalues = trajectory.tensordict["values"][t + 1]

        # note that this is also called the td-error, or delta
        one_step_advantage_estimator = (
            trajectory.tensordict["rewards"][t]
            + gamma * nextvalues * nextnonterminal
            - trajectory.tensordict["values"][t]
        )

        if t == len(advantages) - 1:
            advantages[t] = one_step_advantage_estimator
        else:
            # follows from the bellman equation of the gae, similar to bellman equation of the q function
            advantages[t] = (
                one_step_advantage_estimator + gamma * gae_lambda * advantages[t + 1] * nextnonterminal
            )  # update advantage for current time step

    returns = advantages + trajectory.tensordict["values"]

    return advantages, returns.detach()
