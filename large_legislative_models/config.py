import dataclasses

import numpy as np

""" Principal action range (start, end, num_discretized_rates) inclusive. """
PRINCIPAL_ACTION_SPACES = {
    ("commons_harvest__open", "LLM"): (0, 1, None),
    ("commons_harvest__open", "AID"): (0, 1, None),
    ("commons_harvest__open", "Dual-RL"): (0, 1, 21),
    ("commons_harvest__open", "EpsilonGreedy"): (0, 1, 21),
    ("commons_harvest__open", "ThompsonSampling"): (0, 1, 11),
    ("commons_harvest__open", "UCB"): (0, 1, 11),
    ("commons_harvest__open", "GaussianRegression"): (0, 1, None),
    ("clean_up", "LLM"): (0, 3, None),
    ("clean_up", "AID"): (0, 3, None),
    ("clean_up", "Dual-RL"): (0, 3, 21),
    ("clean_up", "EpsilonGreedy"): (0, 3, 21),
    ("clean_up", "ThompsonSampling"): (0, 3, 11),
    ("clean_up", "UCB"): (0, 3, 11),
    ("clean_up", "GaussianRegression"): (0, 3, None),
    ("cer", "LLM"): (0, 5, None),
    ("cer", "AID"): (0, 5, None),
    ("cer", "Dual-RL"): (0, 5, 6),
    ("cer", "EpsilonGreedy"): (0, 5, 6),
    ("cer", "ThompsonSampling"): (0, 5, 3),
    ("cer", "UCB"): (0, 5, 3),
    ("cer", "GaussianRegression"): (0, 5, None),
}

ENV_CONFIGS = {
    "commons_harvest__open": {"episode_length": 1000, "sampling_horizon": 1000, "num_convergence": 5},
    "clean_up": {"episode_length": 1000, "sampling_horizon": 1000, "num_convergence": 20},
    "cer": {"episode_length": 5, "sampling_horizon": 5, "num_convergence": 2000, "capture_video": False},
}

PRINCIPAL_CONFIGS = {
    "LLM": {
        "temperature": 0.01,
        "llm_model": "gemini-1.5-flash",
        "llm_prompt_style": "prompt_style_1",
    },
    "EpsilonGreedy": {
        "epsilon": 0.1,
    },
    "UCB": {
        "ucb_coef": 0.2,
    },
    "Dual-RL": {
        "principal_ent_coef": 0.2,  # coefficient of entropy loss term for Dual-RL principal
        "principal_update_epochs": 4,  # PPO update epochs for principal
        "principal_clip_coef": 0.2,  # surrogate clipping coefficient for the principal
        "dual_rl_hidden_dim": 128, # MLP hidden layer dimension
    },
    "AID": {
        "aid_hidden_dim": 256, # MLP hidden layer dimension
        "num_convergence": 1, # number of convergence episodes
        "num_validation": 1,
    },
}


@dataclasses.dataclass
class Config:
    """Setup"""

    seed: int = 2  # seed of the experiment
    wandb_project_name: str = "wandb-project"  # wandb project name
    wandb_tags: str = ""  # wandb tags
    log_wandb: bool = False  # if toggled, this experiment will be tracked with Weights and Biases
    log_locally: bool = True  # if toggled, log messages will be printed to stderr
    capture_video: bool = False  # whether to capture videos of the agent performances
    video_freq: int = 10  # capture video every how many principal steps?
    save_model: bool = False  # whether to save model parameters
    save_model_freq: int = 50  # save model parameters every how many principal steps?

    """ Principal """
    principal: str = "EpsilonGreedy"  # which principal to use
    total_principal_steps: int = 10000  # the number of principal steps
    principal_lr: float = 2e-4  # the learning rate of the principal optimizer
    num_val_episodes: int = 1  # number of validation episodes
    principal_gets_historical: bool = True  # whether principal gets historical observations

    """ Env """
    saved_core_path: str = "" # path for saved agent net core ('network')
    saved_heads_path: str = "" # path for saved agent net heads ('actor' and 'critic')
    env_name: str = "commons_harvest__open" # environment name
    cer_fixed_indicator: int = None # fixed lever activation for CER, or chosen randomly if None
    reset_agent_nets: bool = True  # whether to reset agent nets to random initialization
    freeze_agent_net_core: bool = True  # whether to freeze the main body of agent nets
    freeze_whole_agent_net: bool = False # whether to freeze agent nets entirely
    num_parallel_games: int = 1  # the number of parallel game environments

    """ Agent hyperparameters """
    agent_lr: float = 1e-3  # learning rate of the agent optimizer
    minibatch_size: int = 128  # size of minibatches when training
    adam_eps: float = 1e-5  # epsilon value for all adam optimizers
    gamma: float = 0.998  # the discount factor gamma
    gae_lambda: float = 0.98  # the lambda for generalized advantage estimation (GAE)
    agent_update_epochs: int = 4  # number of PPO update epochs
    norm_adv: bool = True  # toggles advantage normalization
    clip_coef: float = 0.2  # PPO clipping coefficient
    value_clip_coef: float = 0.2  # value estimate clipping coefficient
    clip_vloss: bool = True  # toggles whether or not to use a clipped loss for the value function
    agent_ent_coef: float = 0.025  # coefficient of entropy loss term for agents
    vf_coef: float = 0.5  # coefficient of the value function
    principal_vf_coef: float = 0.5  # coefficient of the value function for principal
    max_grad_norm: float = 0.5  # maximum norm for gradient clipping shared by agents and principals
    target_kl: float = None  # the target KL divergence threshold

    """ Likely never changed. """
    log_file: str = None  # the file to log to relative to Globals.LOG_DIR
    cuda: bool = True  # if toggled, cuda will be enabled by default
    wandb_entity: str = "wandb_entity"  # entity (team) of wandb project
    flush_interval: int = 1 # logger flush interval
    algorithm: str = "ppo" # optimiation algorithm

    def __post_init__(self):
        """Set environment and principal configs."""
        env_config = ENV_CONFIGS[self.env_name]
        principal_config = PRINCIPAL_CONFIGS[self.principal]
        self.__dict__.update(env_config)
        self.__dict__.update(principal_config)

        """ Set principal action space upper bound and discretization set. """
        discretization_tuple = PRINCIPAL_ACTION_SPACES.get((self.env_name, self.principal), [None, None, None])
        self.upper_bound = discretization_tuple[1]
        try:
            self.discretization_set = np.linspace(*discretization_tuple).tolist()
        except TypeError:
            pass
