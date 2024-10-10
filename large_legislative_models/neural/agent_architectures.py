import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions.categorical import Categorical


class MeltingPotAgent(nn.Module):
    """
    Agent actor-critic network.
    Must consist of network, actor and critic submodules, of which the former is frozen for AID.
    """

    def __init__(self, observation_shape, num_actions, num_agents, num_brackets):
        super().__init__()
        self.num_brackets = num_brackets
        self.num_actions = num_actions
        self.network = nn.Sequential(
            self.layer_init(nn.Conv2d(observation_shape[2], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = self.layer_init(nn.Linear(512 + num_agents + num_brackets, num_actions), std=0.01)
        self.critic = self.layer_init(nn.Linear(512 + num_agents + num_brackets, 1), std=1)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Layer initialisation."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, obs, player_idx, tax_rates):
        """Provide a value function estimate for an observation.
        Args:
            obs (Tensor): observation
            player_idx (Tensor): one-hot agent indicator
            tax_rates (Tensor): current tax rates
        Returns:
            (Tensor): estimate of the value function
        """

        """ Permute as torch Conv2D needs (N,C,H,W) format. """
        obs_embedding = self.network(obs.permute((0, 3, 1, 2)))

        """ Concatenate agent indicators and current tax rates to convnet output. """
        embedding_with_indicators = torch.cat([obs_embedding, player_idx, tax_rates], dim=-1)

        values = self.critic(embedding_with_indicators).flatten()

        return values

    def generate_action_and_value_no_grads(self, obs, player_idx, tax_rates):
        """Provide an action sampled from policy, its log-probabilities, the policy entropy, and a value function estimate.
        All have gradients detached except the log-probabilities.
        Args:
            obs (Tensor): observation
            player_idx (Tensor): one-hot agent indicator
            tax_rates (Tensor): current tax rates
        Returns:
            tensordict (TensorDict): a tensordict holding results of net forward pass
        """

        """ Permute as torch Conv2D needs (N,C,H,W) format. """
        obs_embedding = self.network(obs.permute((0, 3, 1, 2)))

        """ Concatenate agent indicators and current tax rates to convnet output. """
        embedding_with_indicators = torch.cat([obs_embedding, player_idx, tax_rates], dim=-1)

        logits = self.actor(embedding_with_indicators)
        values = self.critic(embedding_with_indicators).flatten()

        probs = Categorical(logits=logits)
        action = probs.sample()

        tensordict = TensorDict(
            {
                "actions": action.detach(),
                "logprobs": probs.log_prob(action),
                "entropy": probs.entropy().detach(),
                "values": values.detach(),
            }
        )
        return tensordict

    def get_action_logprobs_and_value(self, obs, player_idx, tax_rates, action):
        """Provide policy log-probabilities for a given action and observation, the policy entropy, and a value function estimate.
        All have gradients attached.
        Args:
            obs (Tensor): observation
            player_idx (Tensor): one-hot agent indicator
            tax_rates (Tensor): current tax rates
            action (Tensor): action to provide log-probabilities for
        Returns:
            tensordict (TensorDict): a tensordict holding results of net forward pass
        """

        """ Permute as torch Conv2D needs (N,C,H,W) format. """
        obs_embedding = self.network(obs.permute((0, 3, 1, 2)))

        """ Concatenate agent indicators and current tax rates to convnet output. """
        embedding_with_indicators = torch.cat([obs_embedding, player_idx, tax_rates], dim=-1)

        logits = self.actor(embedding_with_indicators)
        values = self.critic(embedding_with_indicators).flatten()
        probs = Categorical(logits=logits)

        tensordict = TensorDict(
            {
                "actions": action,
                "logprobs": probs.log_prob(action),
                "entropy": probs.entropy(),
                "values": values,
            }
        )
        return tensordict


class CERAgent(nn.Module):
    """
    Agent actor-critic network.
    Must consist of network, actor and critic submodules, of which the former is frozen for AID.
    """

    def __init__(self, observation_shape, num_actions, num_agents, num_brackets):
        super().__init__()
        self.network = nn.Sequential(
            self.layer_init(nn.Linear(observation_shape[0]+num_agents, 64)),
            nn.ReLU(),
            self.layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.actor = self.layer_init(nn.Linear(64, num_actions), std=0.01)
        self.critic = self.layer_init(nn.Linear(64, 1), std=1)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Layer initialisation."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, obs, player_idx, unused):
        """Provide a value function estimate for an observation.
        Args:
            obs (Tensor): observation
            player_idx (Tensor): one-hot agent indicator
        Returns:
            (Tensor): estimate of the value function
        """

        obs_with_indicators = torch.cat([obs, player_idx], dim=-1)

        embedding = self.network(obs_with_indicators)

        values = self.critic(embedding).flatten()

        return values

    def generate_action_and_value_no_grads(self, obs, player_idx, unused):
        """Provide an action sampled from policy, its log-probabilities, the policy entropy, and a value function estimate.
        All have gradients detached except the log-probabilities.
        Args:
            obs (Tensor): observation
            player_idx (Tensor): one-hot agent indicator
        Returns:
            tensordict (TensorDict): a tensordict holding results of net forward pass
        """

        obs_with_indicators = torch.cat([obs, player_idx], dim=-1)

        embedding = self.network(obs_with_indicators)

        logits = self.actor(embedding)
        values = self.critic(embedding).flatten()

        probs = Categorical(logits=logits)
        action = probs.sample()
        tensordict = TensorDict(
            {
                "actions": action.detach(),
                "logprobs": probs.log_prob(action),
                "entropy": probs.entropy().detach(),
                "values": values.detach(),
            }
        )
        return tensordict

    def get_action_logprobs_and_value(self, obs, player_idx, unused, action):
        """Provide policy log-probabilities for a given action and observation, the policy entropy, and a value function estimate.
        All have gradients attached.
        Args:
            obs (Tensor): observation
            player_idx (Tensor): one-hot agent indicator
            action (Tensor): action to provide log-probabilities for
        Returns:
            tensordict (TensorDict): a tensordict holding results of net forward pass
        """

        obs_with_indicators = torch.cat([obs, player_idx], dim=-1)

        embedding = self.network(obs_with_indicators)

        logits = self.actor(embedding)
        values = self.critic(embedding).flatten()

        probs = Categorical(logits=logits)
        tensordict = TensorDict(
            {
                "actions": action,
                "logprobs": probs.log_prob(action),
                "entropy": probs.entropy(),
                "values": values,
            }
        )
        return tensordict


class PrincipalAgent(nn.Module):
    """
    Principal policy network.
    Outputs tax rate suggestions using a discrete action head for each tax bracket.
    """

    def __init__(self, principal_obs_length, num_brackets, discretized_bracket_length, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            self.layer_init(nn.Linear(principal_obs_length, hidden_dim)),
            nn.ReLU(),
            self.layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        """ One action head for each tax bracket, outputting tax suggestions over
        a discretized interval, plus a NO-OP to leave that bracket unchanged. """
        self.actor_heads = nn.ModuleList(
            [self.layer_init(nn.Linear(hidden_dim, discretized_bracket_length+1), std=0.01) for _ in range(num_brackets)]
        )
        self.critic = self.layer_init(nn.Linear(hidden_dim, 1), std=1)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Layer initialisation."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, obs, action=None):
        """If not given an action, provides an action sampled from principal policy,
        its log-probabilities and the policy entropy. If an action is given, provides
        policy log-probabilities for that action and policy entropy.

        Args:
            obs (Tensor): observation
            action (Tensor[num_brackets], optional): action to provide log-probabilities for
                                                     - or if None one is sampled from policy

        Returns:
            tensordict (TensorDict): a tensordict holding results of net forward pass
        """

        hidden = self.mlp(obs)
        logits = [head(hidden) for head in self.actor_heads]
        probs = [Categorical(logits=logit) for logit in logits]
        if action is None:
            action = torch.stack([prob.sample() for prob in probs], dim=1)
        logprobs = sum(prob.log_prob(action[:, i]) for i, prob in enumerate(probs))
        entropy = sum(prob.entropy() for prob in probs)

        values = self.critic(hidden).flatten()
        tensordict = TensorDict(
            {
                "actions": action,
                "logprobs": logprobs,
                "entropy": entropy,
                "values": values,
                "distribution": torch.stack([prob.probs for prob in probs])
            }
        )
        return tensordict


class DesignerNet(nn.Module):
    """ MetaGrad incentive designer network. """

    def __init__(self, principal_obs_length, h_dim, num_brackets, output_multiplier):
        super().__init__()
        self.output_multiplier = output_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(principal_obs_length, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_brackets),
        )

    def forward(self, obs):
        unsigmoided = self.mlp(obs)
        tax_vals = self.output_multiplier * torch.sigmoid(unsigmoided)
        return tax_vals
