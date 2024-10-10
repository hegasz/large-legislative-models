import torch

from tensordict import TensorDict
from large_legislative_models.utils.context import Metrics
from abc import abstractmethod


class BaseAlgorithm:
    @abstractmethod
    def get_policy_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_value_loss(self, *args, **kwargs):
        pass


class PPO(BaseAlgorithm):

    @staticmethod
    def get_policy_loss(
        new_agent_net_outputs: TensorDict,
        trajectory_agent_net_outputs: TensorDict,
        advantages,
        norm_adv: bool,
        clip_coef,
    ): 
        """ Calculate a PPO policy loss and related metrics.

        Args:
            new_agent_net_outputs (TensorDict): forward pass through updated policy net
            trajectory_agent_net_outputs (TensorDict): old policy we are updating from
            advantages (Tensor): advantage estimates
            norm_adv (bool): whether to normalize advantage estimates
            clip_coef (float): PPO clip coefficient

        Returns:
            (Metrics): PPO policy loss and related metrics
        """        

        """ Create class that will store loss metrics. """
        metrics = Metrics()

        """ Add entropy loss to metrics. """
        metrics.entropy_loss = new_agent_net_outputs["entropy"].mean()

        """ Policy ratio required for PPO loss. """
        logratio = new_agent_net_outputs["logprobs"] - trajectory_agent_net_outputs["logprobs"]
        ratio = logratio.exp()

        """ Calculate loss metrics. See http://joschu.net/blog/kl-approx.html. """
        with torch.no_grad():
            metrics.old_approx_kl = (-logratio).mean()
            metrics.approx_kl = ((ratio - 1) - logratio).mean()
            metrics.clipfracs += [
                ((ratio - 1.0).abs() > clip_coef).float().mean().item()
            ]

        """ Normalize advantage estimates. """
        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        """ Form clipped and unclipped objectives. """
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - clip_coef, 1 + clip_coef
        )

        """ Store PPO loss with metrics. """
        metrics.pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        """ Return loss metrics. """
        return metrics

    @staticmethod
    def get_value_loss(
        new_agent_net_outputs: TensorDict,
        trajectory_agent_net_outputs: TensorDict,
        returns,
        clip_vloss,
        clip_coef,
    ):
        """ Calculate a value loss

        Args:
            new_agent_net_outputs (TensorDict): forward pass through updated nets, containing new value estimates
            trajectory_agent_net_outputs (TensorDict): forward pass through nets we are updating from, containing old value estimates
            returns (Tensor): returns estimates to target value estimates to
            clip_vloss (bool): whether to clip value estimates
            clip_coef (float): value loss clip coefficient

        Returns:
            value loss
        """

        """ Form squared differences between value and returns estimates """
        squared_diffs_unclipped = (new_agent_net_outputs["values"] - returns) ** 2

        if not clip_vloss:
            """ Take mean to make this an MSE loss. """
            v_loss = 0.5 * squared_diffs_unclipped.mean()
        else:
            """ Clip new value estimates relative to old value estimates. """
            v_clipped = trajectory_agent_net_outputs["values"] + torch.clamp(
                new_agent_net_outputs["values"] - trajectory_agent_net_outputs["values"],
                -clip_coef,
                clip_coef,
            )
            """ Squared differences between clipped value estimates and returns estimates. """
            squared_diffs_clipped = (v_clipped - returns) ** 2

            """ Take maximum of clipped and unclipped squared differences. """
            squared_diffs = torch.max(squared_diffs_unclipped, squared_diffs_clipped)

            """ Take mean to form MSE over these squared differences. """
            v_loss = 0.5 * squared_diffs.mean()

        """ Return value loss. """
        return v_loss

""" Dictionary of possible update algorithm options. """
algorithm_choices = {
    "ppo": PPO,
}
