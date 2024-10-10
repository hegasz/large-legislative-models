import os
import numpy as np
import torch
import torchvision
from rich import print
from torchopt.base import UninitializedState
from torchopt import pytree
from torchopt.update import apply_updates
from torchopt.typing import TupleOfTensors
from large_legislative_models.globals import Globals


def welfare(rewards):
    """
    Takes in a reward of shape (num_parallel_games, num_agents) and returns tensor of len(num_parallel_games)
    of social welfare of that step across the agents in each game.

    Currently, welfare is the mean.
    """
    return rewards.mean(dim=1)

def initialise_agent_nets(nets, saved_core_path, saved_heads_path, freeze_core, freeze_all):
    """Load pre-trained agents and freeze the "network" module of agent nets."""

    if saved_core_path != "":
        print(f"[bold blue]Loading agent net core from {saved_core_path}[/]")
        saved_whole_model = torch.load(saved_core_path)
        saved_core = {k: v for k, v in saved_whole_model.items() if k.startswith("network.")}
        nets.load_state_dict(saved_core, strict=False)

    if saved_heads_path != "":
        print(f"[bold blue]Loading agent heads from {saved_core_path}[/]")
        saved_whole_model = torch.load(saved_heads_path)
        saved_heads = {
            k: v for k, v in saved_whole_model.items() if (k.startswith("actor.") or k.startswith("critic."))
        }
        nets.load_state_dict(saved_heads, strict=False)

    if freeze_core:
        print(f"[bold blue]Freezing agent net core[/]")
        for param in nets.network.parameters():
            param.requires_grad = False

    if freeze_all:
        print(f"[bold blue]Freezing whole agent net[/]")
        for param in nets.parameters():
            param.requires_grad = False

def capture_video(frames, episode_number):
    video = torch.from_numpy(np.stack(frames))
    torchvision.io.write_video(
        f"{Globals.LOG_DIR}/episode{episode_number}.mp4",
        video,
        fps=100,
        video_codec="vp9",
    )
    print(f"Video tensor shape: {video.shape}")
    print(f"Video tensor min: {video.min()}, max: {video.max()}")
    print(f"Video saved for episode {episode_number}")
    return f"{Globals.LOG_DIR}/episode{episode_number}.mp4"

def get_flush(step, flush_interval):
    if step // flush_interval == 0:
        return True
    return False

def make_save_dir(episode_number):
    """Create directory we save to.

    Args:
        episode_number (int): current episode number
    """

    try:
        os.mkdir(f"./saved_params")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"./saved_params/step{episode_number}")
    except FileExistsError:
        pass

def format_returns(cumulative_rewards, num_agents, prefix):
    rewards = cumulative_rewards.view(-1, num_agents)
    result = {f"{prefix}game {i} mean return": val for i, val in enumerate(rewards.mean(dim=1))}
    # for i in range(len(rewards)):
    #     for j, val in enumerate(rewards[i]):
    #         result[f"{prefix}game {i} agent {j} return"] = val
    return result

def format_principal_returns(principal_rewards, prefix):
    result = {f"{prefix}game {i} principal return": val for i, val in enumerate(principal_rewards)}
    return result

def format_taxes(tax_vals_per_game, prefix):
    tax_vals = {}
    for i in range(len(tax_vals_per_game)):
        for bracket, val in enumerate(tax_vals_per_game[i]):
            tax_vals[f"{prefix}game {i} tax bracket {bracket+1}"] = val
    return tax_vals

def mod_step(self, loss: torch.Tensor, max_grad_norm, clip_grads=True):
    """Compute the gradients of the loss to the network parameters and update network parameters.

    Graph of the derivative will be constructed, allowing to compute higher order derivative
    products. We use the differentiable optimizer (pass argument ``inplace=False``) to scale the
    gradients and update the network parameters without modifying tensors in-place.

    Args:
        loss (torch.Tensor): The loss that is used to compute the gradients to the network
            parameters.
    """
    # Step parameter only
    for i, (param_container, state) in enumerate(
        zip(self.param_containers_groups, self.state_groups),
    ):
        flat_params: TupleOfTensors
        flat_params, container_treespec = pytree.tree_flatten_as_tuple(param_container)  # type: ignore[arg-type]
        if isinstance(state, UninitializedState):
            state = self.impl.init(flat_params)
        grads = torch.autograd.grad(
            loss,
            flat_params,
            create_graph=True,
            allow_unused=True,
        )

        if clip_grads:
            grad_vector = torch.cat([grad.view(-1) for grad in grads if grad is not None])
            total_norm = torch.norm(grad_vector, p=2)
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            # clamping can be replaced by an (if < 1) but torch source code cites reasons against.
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            clipped_grads = tuple([clip_coef_clamped * grad if grad is not None else None for grad in grads])

            updates, new_state = self.impl.update(
                clipped_grads,
                state,
                params=flat_params,
                inplace=False,
            )
        else:
            updates, new_state = self.impl.update(
                grads,
                state,
                params=flat_params,
                inplace=False,
            )

        self.state_groups[i] = new_state
        flat_new_params = apply_updates(flat_params, updates, inplace=False)

        new_params: ModuleTensorContainers = pytree.tree_unflatten(  # type: ignore[assignment]
            container_treespec,
            flat_new_params,
        )
        for container, new_param in zip(param_container, new_params):
            container.update(new_param)
