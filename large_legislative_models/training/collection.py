import os

import torch
from tensordict import TensorDict

from large_legislative_models.training.optimize import step_agent_nets
from large_legislative_models.utils import capture_video, format_returns
from large_legislative_models.utils.buffer import TrainingEpisode, ValidationEpisode
from large_legislative_models.utils.logger import logger

dir = os.getcwd()


def run_training_episode(ctx, envs, args, agent_trajectory, tax_vals_per_game, log_prefix="train/"):
    """Collect a training episode, stepping agent nets after every fixed length trajectory.

    Args:
        ctx (Context): training context
        envs: game environments
        args (Config): configuration arguments
        agent_trajectory (FixedLengthTrajectory): agent trajectory buffer we reuse to save memory
        tax_vals_per_game (Tensor[num_parallel_games, num_brackets]): tax values set for this episode

    Returns:
        episode_buffer (TrainingEpisode): filled buffer for this episode
    """

    """ Reset context for a new training episode. """
    ctx.new_episode()

    """ Initialise episode buffer """
    episode_buffer = TrainingEpisode(
        agent_trajectory=agent_trajectory,
        envs=envs,
        num_parallel_games=args.num_parallel_games,
        episode_length=args.episode_length,
        tax_rates_for_this_ep=tax_vals_per_game,
        record_world_obs=(args.capture_video and (ctx.episode_number + 1) % args.video_freq == 0),
    )

    """ Step through episode collection in sampling_horizon length chunks. """
    for _ in range(ctx.num_data_collect_per_ep):
        """Collect a trajectory."""
        collect_fixed_length_trajectory(
            ctx=ctx,
            envs=envs,
            sampling_horizon=args.sampling_horizon,
            episode_buffer=episode_buffer,
            keep_log_prob_grads=False,
        )
        """ Step agent nets on this trajectory. """
        step_agent_nets(args, ctx, episode_buffer.agent_trajectory, tax_vals_per_game)

    """ Save and log video for the first parallel game. """
    if args.capture_video and (ctx.episode_number + 1) % args.video_freq == 0:
        path = capture_video(episode_buffer.first_parallel_game_world_obs, ctx.episode_number)
        logger.log_video(path, f"episode{ctx.episode_number}")

    """ Log episode metrics. """
    logger.log_later(
        {
            "combined_val_train/episode": ctx.total_episode_number,
            "train/episode": ctx.episode_number,
            **format_returns(episode_buffer.logging_agent_cumulative_rewards, ctx.num_agents, prefix=log_prefix),
            **episode_buffer.get_formatted_infos(prefix=log_prefix),
        },
        flush=True,
    )

    """ Return filled episode buffer. """
    return episode_buffer


def run_validation_episode(
    ctx,
    envs,
    num_parallel_games,
    episode_length,
    sampling_horizon,
    tax_vals_per_game,
    keep_log_prob_grads,
    log_prefix="validation/",
):
    """Collect a validation episode without stepping agent nets.

    Args:
        ctx (Context): training context
        envs: game environments
        num_parallel_games (int): number of parallel games
        episode_length (int): episode length
        sampling_horizon (int): length of sampling horizon we update from (subset of episode)
        tax_vals_per_game (Tensor[num_parallel_games, num_brackets]): tax values set for this episode
        keep_log_prob_grads (bool): whether to keep gradients on agent log-probabilities (True for meta-gradient methods)

    Returns:
        episode_buffer (ValidationEpsiode): filled buffer for this episode
    """

    """ Reset context for a new validation episode. """
    ctx.new_validation_episode()

    """ Initialise episode buffer """
    episode_buffer = ValidationEpisode(
        envs=envs,
        num_parallel_games=num_parallel_games,
        episode_length=episode_length,
        tax_rates_for_this_ep=tax_vals_per_game,
        record_world_obs=False,
    )

    """ Step through episode collection in sampling_horizon length chunks. """
    for _ in range(ctx.num_data_collect_per_ep):
        """Collect a trajectory."""
        collect_fixed_length_trajectory(
            ctx=ctx,
            envs=envs,
            sampling_horizon=sampling_horizon,
            episode_buffer=episode_buffer,
            keep_log_prob_grads=keep_log_prob_grads,
        )

    """ Log principal reward for this validation episode. """
    logger.log_later(
        {
            "combined_val_train/episode": ctx.total_episode_number,
            "validation/episode": ctx.validation_episode_number,
            **format_returns(episode_buffer.logging_agent_cumulative_rewards, ctx.num_agents, prefix=log_prefix),
            **episode_buffer.get_formatted_infos(prefix=log_prefix),
        },
        flush=True,
    )

    """ Return filled episode buffer. """
    return episode_buffer


def collect_fixed_length_trajectory(
    ctx,
    envs,
    sampling_horizon,
    episode_buffer,
    keep_log_prob_grads,
):
    """Collect a fixed length trajectory of agent experience.

    Args:
        ctx (Context): the context
        envs: parallel game environments
        sampling_horizon (int): length of trajectory to collect
        episode_buffer (EpisodeBuffer): buffer for episode this trajectory belongs to
        keep_log_prob_grads (bool): whether to keep gradient flow on log-probabilities
    """

    """ Clear gradients in continually re-used trajectory. """
    episode_buffer.reset_agent_trajectory()

    """ Loop through trajectory steps. """
    for trajectory_step in range(0, sampling_horizon):
        """Retrieve observation and done stored at previous loop iteration."""
        cached_observation, cached_done = episode_buffer.get_cached()

        """ Player indices and tax rates for this step, to use in net forward pass. """
        step_player_indices = torch.arange(ctx.num_agents).repeat(episode_buffer.num_parallel_games)
        step_player_indices_one_hot = torch.nn.functional.one_hot(step_player_indices, num_classes=ctx.num_agents)
        step_game_indices = torch.arange(episode_buffer.num_parallel_games).repeat_interleave(ctx.num_agents)
        step_tax_vals = episode_buffer.tax_rates_for_this_ep[step_game_indices]

        """ For meta-gradients, we want to keep gradients on log_probs - else no gradients at all. """
        with torch.set_grad_enabled(keep_log_prob_grads):
            """Produce agent net action and value for this step."""
            agent_step_data: TensorDict = ctx.agent.generate_action_and_value_no_grads(
                cached_observation.to(ctx.device),
                step_player_indices_one_hot.to(ctx.device),
                step_tax_vals.to(ctx.device),
            ).cpu()

        """ Record observation and done produced at last iteration in this step. """
        agent_step_data["obs"] = cached_observation
        agent_step_data["dones"] = cached_done

        """ Step environment. """
        agent_next_obs, agent_reward, terminations, truncations, info = envs.step(agent_step_data["actions"].cpu().numpy())
        agent_next_done = torch.logical_or(terminations, truncations)

        """ Record rewards for this step. """
        agent_step_data["rewards"] = agent_reward

        """ Send data from this step to the buffer. """
        episode_buffer.record(agent_step_data, info, trajectory_step, ctx.episode_step)

        """ Set observation and done separately to reward etc as they belong to the NEXT timestep and will be recorded then. """
        episode_buffer.cache_next(
            agent_next_done=agent_next_done,
            agent_next_obs=agent_next_obs.float(),
        )

        """ (optional) Log step information. """
        # logger.log_later(
        #     {
        #         # **format_returns(
        #         #     episode_buffer.logging_agent_cumulative_rewards, ctx.num_agents, prefix="within_episode/"
        #         # ),
        #         **episode_buffer.get_formatted_infos(prefix="within_episode/"),
        #     },
        #     flush=True,
        # )

        """ Increment step for episode this trajectory is part of. """
        ctx.episode_step += 1
