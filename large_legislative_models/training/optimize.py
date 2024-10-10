import numpy as np
import torch
from tensordict import TensorDict

from large_legislative_models.config import Config
from large_legislative_models.utils.buffer import FixedLengthTrajectory
from large_legislative_models.utils.context import Context, Metrics
from large_legislative_models.utils.logger import logger
from large_legislative_models.utils.loss_calculation_utils import gae_advantage_and_return_estimates
import torch.nn as nn

def step_agent_nets(args: Config, ctx: Context, trajectory: FixedLengthTrajectory, tax_vals_per_game):
    """ Step agent nets using a fixed length sampling trajectory. """

    """ Get value and status of very last observation in trajectory. """
    step_player_indices = torch.arange(ctx.num_agents).repeat(args.num_parallel_games)
    step_player_indices_one_hot = torch.nn.functional.one_hot(step_player_indices, num_classes=ctx.num_agents)
    step_game_indices = torch.arange(args.num_parallel_games).repeat_interleave(ctx.num_agents)
    step_tax_vals = tax_vals_per_game[step_game_indices]
    with torch.no_grad():
        next_value = ctx.agent.get_value(trajectory.next_obs.to(ctx.device), step_player_indices_one_hot.to(ctx.device), step_tax_vals.to(ctx.device)).cpu()
    next_done = trajectory.next_done

    """ Produce advantage and returns estimates. Gradient kept on advantages (flows through rewards) but detached on returns. """
    advantages, returns = gae_advantage_and_return_estimates(trajectory, next_done, next_value, args.gamma, args.gae_lambda)

    """ Flatten advantage and return estimates. """
    returns = returns.reshape(-1)
    advantages = advantages.reshape(-1)

    """ Array of indices for the batch. """
    flattened_trajectory_indices = np.arange(len(returns))

    """ Keep track of player indices and tax rates in that parallel game for all batch indices. """
    flattened_trajectory_player_idx = torch.arange(ctx.num_agents).repeat((trajectory.shared_shape[0],args.num_parallel_games)).reshape(-1)
    flattened_trajectory_game_idx = torch.arange(args.num_parallel_games).repeat_interleave(ctx.num_agents).repeat((trajectory.shared_shape[0],1)).reshape(-1)
    flattened_trajectory_tax_vals = tax_vals_per_game[flattened_trajectory_game_idx]

    """ Loop over desired number of update epochs on this sampled trajectory. """
    for epoch in range(args.agent_update_epochs):

        """ Shuffle batch indices. """
        np.random.shuffle(flattened_trajectory_indices)

        """ Proceed through shuffled batch stepping nets over minibatches. """
        for start in range(0, len(flattened_trajectory_indices), args.minibatch_size):

            """ Form minibatch. """
            end = start + args.minibatch_size
            minibatch_indices = flattened_trajectory_indices[start:end]
            trajectory_batch = trajectory.get_batch(minibatch_indices)

            """ Player indices and tax rates for this minibatch. """
            batch_player_indices = flattened_trajectory_player_idx[minibatch_indices]
            # batch_player_indices_one_hot has shape (minibatch_size, num_agents)
            batch_player_indices_one_hot = torch.nn.functional.one_hot(batch_player_indices, num_classes=ctx.num_agents)
            batch_tax_rates = flattened_trajectory_tax_vals[minibatch_indices]

            """ Net outputs for this minibatch. """
            new_agent_net_outputs: TensorDict = ctx.agent.get_action_logprobs_and_value(
                trajectory_batch["obs"],
                batch_player_indices_one_hot.to(ctx.device),
                batch_tax_rates.to(ctx.device),
                trajectory_batch["actions"],
            )

            """ Compute policy loss and training metrics for this minibatch. """
            metrics: Metrics = ctx.alg.get_policy_loss(
                new_agent_net_outputs=new_agent_net_outputs,
                trajectory_agent_net_outputs=trajectory_batch,
                advantages=advantages[minibatch_indices].to(ctx.device),
                norm_adv=args.norm_adv,
                clip_coef=args.clip_coef,
            )

            """ Compute value loss for this minibatch. """
            metrics.v_loss = ctx.alg.get_value_loss(
                new_agent_net_outputs=new_agent_net_outputs,
                trajectory_agent_net_outputs=trajectory_batch,
                returns=returns[minibatch_indices].to(ctx.device),
                clip_vloss=args.clip_vloss,
                clip_coef=args.value_clip_coef,
            )

            """ Step agent parameters - differentiably for AID. """
            if args.principal == "AID":
                differentiable_agent_step(metrics, ctx.agent, ctx.agent_actor_opt, ctx.agent_critic_opt, args)
            else:
                agent_step(metrics, ctx.agent, ctx.agent_opt, args)

        """ Break update epoch loop if approximate KL divergence gets too large. """
        if args.target_kl is not None:
            if metrics.approx_kl > args.target_kl:
                break

        """ Log loss metrics. """
        logger.log_later(
            {
                "opt/step": (ctx.episode_number - 1) * args.agent_update_epochs + epoch,
                f"opt/policy_update_epoch": epoch,
                f"opt/agent_pg_loss": metrics.pg_loss,
                f"opt/agent_v_loss": metrics.v_loss,
                f"opt/agent_entropy_loss": metrics.entropy_loss,
                f"opt/agent_approx_kl": metrics.approx_kl,
                f"opt/agent_clipfrac": np.mean(metrics.clipfracs),
            },
            flush=True,
        )

def agent_step(metrics, agent_nets, agent_opt, args):
    """ Vanilla PPO agent net step with non-differentiable torch optimizer. """

    """ Form loss. """
    loss = metrics.pg_loss - args.agent_ent_coef * metrics.entropy_loss + metrics.v_loss * args.vf_coef

    """ Step agent nets. """
    agent_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent_nets.parameters(), args.max_grad_norm)
    agent_opt.step()


def differentiable_agent_step(metrics, agent_nets, agent_actor_opt, agent_critic_opt, args):
    """Differentiable agent step method for MetaGrad.
    Actor head is trained with a differentiable optimizer.
    Critic head is trained with a regular pytorch optimizer.
    "network" submodule, the rest of the agent net, is assumed to be frozen.
    """

    """ Form actor and critic components of PPO loss separately. """
    actor_loss = metrics.pg_loss - args.agent_ent_coef * metrics.entropy_loss
    critic_loss = metrics.v_loss * args.vf_coef

    """ Step actor head. """
    agent_actor_opt.step(agent_actor_opt, loss=actor_loss, max_grad_norm=args.max_grad_norm)

    """ Step critic head. """
    agent_critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(agent_nets.critic.parameters(), args.max_grad_norm)
    agent_critic_opt.step()


def step_principal_agent_nets(principal_agent, optimizer, principal_rewards_per_game, tax_decision_data, ctx, args):
    """ Step principal agent nets on one-step trajectory. """

    """ For one-step trajectory, Q-function is rewards so advantage calculation is simplified. """
    advantage_estimates = principal_rewards_per_game.to(ctx.device) - tax_decision_data["values"]
    returns_estimates = principal_rewards_per_game.to(ctx.device)

    """ Loop over desired number of update epochs on this one-step trajectory. """
    for epoch in range(args.principal_update_epochs):
        """New policy net outputs, given original sigma observation and action from this one-step trajectory."""
        new_net_outputs: TensorDict = principal_agent(
            tax_decision_data["obs"],
            tax_decision_data["actions"],  # already on device from when it was generated
        )

        """ Compute policy loss and training metrics. """
        metrics: Metrics = ctx.alg.get_policy_loss(
            new_agent_net_outputs=new_net_outputs,
            trajectory_agent_net_outputs=tax_decision_data,
            advantages=advantage_estimates.to(ctx.device),
            norm_adv=False,
            clip_coef=args.principal_clip_coef,
        )

        """ Compute value loss. """
        metrics.v_loss = ctx.alg.get_value_loss(
            new_agent_net_outputs=new_net_outputs,
            trajectory_agent_net_outputs=tax_decision_data,
            returns=returns_estimates,
            clip_vloss=args.clip_vloss,
            clip_coef=args.value_clip_coef,
        )

        """ Form PPO loss. """
        loss = metrics.pg_loss - args.principal_ent_coef * metrics.entropy_loss + args.principal_vf_coef * metrics.v_loss

        """ Step policy net. """
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(principal_agent.parameters(), args.max_grad_norm)
        optimizer.step()
        ctx.principal_opt_step += 1

        """ (optional) Logging distributions: useful for debugging, but large logging overhead. """
        # logger.log_distribution(new_net_outputs["distribution"][0][0],0,principal_step, epoch)

        """ Log loss metrics. """
        logger.log_later(
            {
                "principal_opt/step": ctx.principal_opt_step,
                f"principal_opt/policy_update_epoch": epoch,
                f"principal_opt/pg_loss": metrics.pg_loss,
                f"principal_opt/value_estimate": tax_decision_data["values"][0].item(),
                f"principal_opt/entropy_loss": metrics.entropy_loss,
                f"principal_opt/approx_kl": metrics.approx_kl,
                f"principal_opt/clipfrac": np.mean(metrics.clipfracs),
            },
            flush=True,
        )
