import torch
import torchopt
from large_legislative_models.utils import format_principal_returns, format_taxes, make_save_dir
import wandb
from rich import print

from large_legislative_models import setup_config_and_run_dir, setup_experiment
from large_legislative_models.config import Config
from large_legislative_models.training.collection import run_training_episode, run_validation_episode
from large_legislative_models.utils.logger import logger

def main():
    args: Config = setup_config_and_run_dir()
    print(args)
    logger.cfg(args)

    ctx, envs, agent_trajectory = setup_experiment(args)

    for principal_step in range(args.total_principal_steps):

        """ Set environment indicators. Return one-hot indicators for each game, or (num_games, 0) empty tensor. """
        indicator_per_game = envs.set_indicator()

        """ If differentiable optimizers are being used, their gradients should be stopped at each outer loop iteration. """
        try:
            torchopt.stop_gradient(ctx.agent.actor)
            torchopt.stop_gradient(ctx.agent_actor_opt)
        except AttributeError:
            pass

        """ Decide new tax rates / incentives and return them. """
        gradient_hot_tax_vals_per_game = ctx.principal.set_tax_vals(indicator_per_game, principal_step)

        """ Set chosen tax rates / incentives in environments. """
        envs.apply_principal_action(gradient_hot_tax_vals_per_game)

        """ From hereon, utmost care must be taken with gradient flows.
        Tax rates / incentives directly output by principals are termed "gradient-hot"
        as they may have gradients flowing through them - which e.g. should not flow through
        the tax rate indicators agents later observe. """
        detached_tax_vals_per_game = gradient_hot_tax_vals_per_game.clone().detach()

        """ Reset agent nets to their orginal parameters. """
        if args.reset_agent_nets:
            ctx.reset_agent()

        """ Convergence episodes. """
        for _ in range(args.num_convergence):
            """ Run a training episode - collect trajectories and step agent nets in chunks of sampling_horizon. """
            _ = run_training_episode(
                ctx, envs, args, agent_trajectory, detached_tax_vals_per_game
            )

        """ Validation episodes. """
        episode_buffers = [
            run_validation_episode(
                ctx,
                envs,
                args.num_parallel_games,
                args.episode_length,
                args.sampling_horizon,
                detached_tax_vals_per_game,
                keep_log_prob_grads=args.principal=="AID",
            )
            for _ in range(args.num_val_episodes)
        ]

        """ Log principal returns and tax rates set. """
        logger.log_later(
            {
                "combined_val_train/episode": ctx.total_episode_number,
                "principal_final/principal_step": principal_step,
                **format_principal_returns(
                    torch.stack([ep.principal_cumulative_reward for ep in episode_buffers]).mean(dim=0),
                    prefix="principal_final/",
                ),
                **format_taxes(detached_tax_vals_per_game, prefix="principal_final/"),
            },
            flush=True,
        )

        """ Update principal. """
        ctx.principal.update(ctx, principal_step, episode_buffers)

        """ Save model parameters. """
        if args.save_model and principal_step % args.save_model_freq == 0:
            make_save_dir(principal_step)
            torch.save(
                ctx.agent.state_dict(),
                f"./saved_params/step{principal_step}/seed_{args.seed}_agent_net.pt",
            )
            ctx.principal.save_params(principal_step)

    wandb.finish()


if __name__ == "__main__":
    main()
