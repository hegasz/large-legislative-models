from copy import deepcopy
from typing import List
from dataclasses import dataclass, field
import torch


@dataclass
class Metrics:
    old_approx_kl: torch.Tensor = None
    approx_kl: torch.Tensor = None
    clipfracs: List[float] = field(default_factory=list)
    pg_loss: float = -1
    v_loss: float = -1
    entropy_loss: float = -1


class Context:
    def __init__(self, args, num_agents, device, agent, alg):
        self.num_agents = num_agents
        self.device = device
        self.agent = agent
        self.original_agent_nets = deepcopy(self.agent.state_dict())
        self.alg = alg

        # iteration step trackers
        self.num_data_collect_per_ep = args.episode_length // args.sampling_horizon
        self.episode_step = 0
        self.episode_number = 0  # This just refers to training episodes
        self.validation_episode_number = 0
        self.total_episode_number = 0
        self.principal_opt_step = 0

    def new_validation_episode(self):
        self.validation_episode_number += 1
        self.total_episode_number += 1
        self.episode_step = 0

    def new_episode(self):
        """ No need to reset obs, actions, logprobs, etc as they have length args.sampling_horizon
        and will be overwitten in the agent trajectory buffer. """
        self.episode_number += 1
        self.total_episode_number += 1
        self.episode_step = 0

    def reset_agent(self):
        self.agent.load_state_dict({name: self.original_agent_nets[name] for name in self.original_agent_nets})
