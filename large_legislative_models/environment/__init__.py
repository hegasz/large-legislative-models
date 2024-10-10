import torch

from large_legislative_models.config import Config
from large_legislative_models.environment.vector_envs import (
    MarkovVectorEnv,
    ParallelEnv,
    create_parallel_games,
)

from .transforms import IncentiveTransform, TaxTransform


def create_meltingpot_envs(args: Config, env_config):
    env = parallel_env = ParallelEnv(env_config=env_config, max_steps=args.episode_length)
    env.render_mode = "rgb_array"

    """ Add tax / incentive wrappers. """
    if args.env_name == "clean_up":
        env = IncentiveTransform(env, initial_num_apples=122)
    elif args.env_name == "commons_harvest__open":
        env = TaxTransform(env)

    """ Wraps multiple agent envs into a parallel game env. """
    env = MarkovVectorEnv(env)

    """ Wraps multiple parallel game envs into a combined env. """
    # if only one parallel game, no multiprocessing will be used, obviously.
    envs = create_parallel_games(env, num_vec_envs=args.num_parallel_games, multiprocessing=True)

    envs.agent_observation_shape = parallel_env.observation_space(0)["RGB"].shape
    envs.num_actions = parallel_env.action_space(0).n
    envs.principal_action_length = 3
    envs.action_space_shape = ()
    envs.num_agents = parallel_env.max_num_agents
    envs.set_indicator = lambda: torch.empty(args.num_parallel_games, 0)
    envs.name = args.env_name
    return envs
