"""SED LLM"""

"""
Other global variables
"""
import os
import time
from argparse import Namespace
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch
import torch.optim as optim
from art import tprint
from dotenv import load_dotenv
from eztils import datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from eztils.torch import seed_everything
from rich import print
from builtins import print as plain_print

import torchopt

from large_legislative_models.config import Config
from large_legislative_models.environment import create_meltingpot_envs
from large_legislative_models.environment.cer import ParallelCER
from large_legislative_models.globals import Globals
from large_legislative_models.principal.bandit_principals import (
    UCB,
    BanditWrapper,
    EpsilonGreedy,
    ThompsonSampling,
)
from large_legislative_models.principal.basic_principals import FixedTaxRate
from large_legislative_models.principal.bayesian_principals import GaussianRegression
from large_legislative_models.principal.core_principals import Designer, DualRLPrincipal, LLMPrincipal
from large_legislative_models.training.algorithms import BaseAlgorithm, algorithm_choices
from large_legislative_models.utils.buffer import FixedLengthTrajectory
from large_legislative_models.utils.context import Context
from meltingpot import substrate

s = time.time()
from large_legislative_models.utils import initialise_agent_nets, mod_step

load_dotenv()


def get_version() -> str:
    try:
        return importlib_metadata.version("large_legislative_models")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version


def setup_config_and_run_dir() -> Config:
    """
    Sets up the experiment by creating a run directory and a log directory, and creating a symlink from the repo directory to the run directory.
    """
    print("Setting up experiment...")

    # create run dir
    Globals.RUN_DIR = setup_path(Globals.DATA_ROOT / "runs")
    Globals.LOG_DIR = setup_path(Globals.RUN_DIR / datestr())

    print(f"LOG DIR: {Globals.LOG_DIR}")

    """ Symlink repo dir / runs to run_dir. """
    if not (Globals.REPO_DIR / "runs").exists() and (Globals.REPO_DIR / "runs") != Globals.RUN_DIR:
        print(f'Creating symlink from {Globals.REPO_DIR / "runs"} to {Globals.RUN_DIR}')
        (Globals.REPO_DIR / "runs").symlink_to(Globals.RUN_DIR)

    os.chdir(Globals.LOG_DIR)

    """SETUP CONFIG"""
    parser = HfArgumentParser(Config)
    parser.add_argument("-c", "--config", type=str)

    conf: Config
    extras: Namespace
    conf, extras = parser.parse_args_into_dataclasses()

    if extras.config is not None:  # parse config file
        config_path = Path(extras.config)
        if not config_path.is_file():
            print(f"config file {config_path} not found. CWD: {os.getcwd()}")
        (original_conf,) = parser.parse_json_file(extras.config)
        conf = update_dataclass_defaults(Config, original_conf)
        # reinit the parser so that the command line args overwrite the file-specified args
        parser = HfArgumentParser(update_dataclass_defaults(Config, original_conf))
        parser.add_argument("-c", "--config", type=str)
        conf, extras = parser.parse_args_into_dataclasses()

    parser.to_json([conf], Globals.LOG_DIR / "config.json")
    assert (
        conf.episode_length % conf.sampling_horizon == 0
    ), f"conf.episode_length ({conf.episode_length}) must be divisible by conf.sampling_horizon ({conf.sampling_horizon}) without a remainder."
    """CWD is in runs/some_timestamp, we need to go back two levels to get to frozen nets"""
    pardir = os.path.dirname(os.path.dirname(os.getcwd()))

    if conf.saved_core_path != "":
        conf.saved_core_path = pardir + conf.saved_core_path
    else:
        conf.saved_core_path = ""
    if conf.saved_heads_path != "":
        conf.saved_heads_path = pardir + conf.saved_heads_path
    else:
        conf.saved_heads_path = ""

    return conf


def setup_experiment(args: Config):
    """-----------Welcome printing-----------"""
    print(f"[bold green]Welcome to[/]")
    with open(f"{Globals.REPO_DIR}/title.txt", "r") as f:
        plain_print(f.read())
    print(f"[bold green]v{version}[/]")
    tprint(args.principal)

    """ -----------Torch setup----------- """
    torch.set_num_threads(1)
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"[bold purple]Device: {device}[/]")

    """ -----------Env setup----------- """
    if args.env_name in ["clean_up", "commons_harvest__open"]:
        envs = create_meltingpot_envs(args, substrate.get_config(args.env_name))
        from large_legislative_models.neural.agent_architectures import MeltingPotAgent as Agent
    elif args.env_name == "cer":
        envs = ParallelCER(
            num_parallel_games=args.num_parallel_games,
            num_agents=5,
            min_at_lever=2,
            num_levers=3,
            max_steps=args.episode_length,
            fixed_indicator=args.cer_fixed_indicator,
        )
        from large_legislative_models.neural.agent_architectures import CERAgent as Agent

    """ -----------Player agent setup----------- """
    agent = Agent(
        envs.agent_observation_shape,
        envs.num_actions,
        envs.num_agents,
        envs.principal_action_length,
    ).to(device)
    initialise_agent_nets(
        nets=agent,
        saved_core_path=args.saved_core_path,
        saved_heads_path=args.saved_heads_path,
        freeze_core=args.freeze_agent_net_core,
        freeze_all=args.freeze_whole_agent_net,
    )
    # to save memory, we use one small trajectory that will be continually overwritten
    agent_trajectory = FixedLengthTrajectory(
        trajectory_length=args.sampling_horizon,
        base_shape=args.num_parallel_games * envs.num_agents,
        obs_shape=envs.agent_observation_shape,
        action_shape=envs.action_space_shape,
        device=device,
    )

    """ -----------Context setup----------- """
    alg: BaseAlgorithm = algorithm_choices[args.algorithm]
    ctx = Context(
        args=args,
        num_agents=envs.num_agents,
        device=device,
        agent=agent,
        alg=alg,
    )

    """ -----------Optimizer setup----------- """
    if args.principal == "AID":
        # differentiable optimizer for actor and regular optimizer for critic
        agent_actor_opt = torchopt.MetaAdam(
            agent.actor,
            lr=args.agent_lr,
            eps=args.adam_eps,
            use_accelerated_op=True,
        )
        agent_actor_opt.step = mod_step
        agent_critic_opt = optim.Adam(
            agent.critic.parameters(),
            lr=args.agent_lr,
            eps=args.adam_eps,
        )
        ctx.agent_actor_opt = agent_actor_opt
        ctx.agent_critic_opt = agent_critic_opt
    else:
        # regular torch optimizer
        agent_opt = optim.Adam(agent.parameters(), lr=args.agent_lr, eps=args.adam_eps)
        ctx.agent_opt = agent_opt

    """ -----------Principal setup----------- """
    if args.env_name == "commons_harvest__open":
        principal_obs_length = args.episode_length
        num_indicators = 1  # number of contexts for bandit algorithms
    elif args.env_name == "clean_up":
        principal_obs_length = args.episode_length
        num_indicators = 1
    elif args.env_name == "cer":
        principal_obs_length = envs.num_levers + 6
        num_indicators = envs.num_levers
    principal_obs_length = (
        args.episode_length if args.env_name in ["commons_harvest__open", "clean_up"] else envs.num_levers + 6
    )
    if args.principal == "AID":
        ctx.principal = Designer(args, envs.num_agents, device, principal_obs_length, envs.principal_action_length)
    elif args.principal == "Dual-RL":
        ctx.principal = DualRLPrincipal(
            args, envs.num_agents, device, principal_obs_length, envs.principal_action_length
        )
    elif args.principal == "LLM":
        ctx.principal = LLMPrincipal(args, envs.principal_action_length, envs)
    elif args.principal[:5] == "Fixed":
        # set as e.g. "principal = Fixed-[0.4,0.1,0.9]" in config
        fixed_rates = [float(rate) for rate in args.principal[7:-1].split(",")]
        ctx.principal = FixedTaxRate(args, envs, fixed_rates)
    elif args.principal == "EpsilonGreedy":
        bandit = EpsilonGreedy(
            num_indicators=num_indicators,
            arm_count=len(args.discretization_set) ** envs.principal_action_length,
            epsilon=args.epsilon,
            seed=args.seed,
        )
        ctx.principal = BanditWrapper(args, bandit, envs.principal_action_length)
    elif args.principal == "UCB":
        bandit = UCB(
            num_indicators=num_indicators,
            arm_count=len(args.discretization_set) ** envs.principal_action_length,
            coef=args.ucb_coef,
            seed=args.seed,
        )
        ctx.principal = BanditWrapper(args, bandit, envs.principal_action_length)
    elif args.principal == "ThompsonSampling":
        bandit = ThompsonSampling(
            num_indicators=num_indicators,
            arm_count=len(args.discretization_set) ** envs.principal_action_length,
            seed=args.seed,
        )
        ctx.principal = BanditWrapper(args, bandit, envs.principal_action_length)
    elif args.principal == "GaussianRegression":
        ctx.principal = GaussianRegression(agent, args)
    else:
        raise NotImplementedError

    return ctx, envs, agent_trajectory
