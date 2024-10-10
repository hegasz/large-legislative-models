from typing import Any, List, Optional

import cloudpickle
import gymnasium
import gymnasium.vector
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Space
from gymnasium import Env
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
import numpy as np
import torch
from gymnasium.spaces import Discrete, Box
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
import functools
from gymnasium import utils as gym_utils
from matplotlib import pyplot as plt
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from meltingpot import substrate
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
import multiprocessing as mp
import sys
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import logger
from gymnasium.core import Env, ObsType
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    clear_mpi_env_vars,
    concatenate,
    create_empty_array,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gymnasium.vector.vector_env import VectorEnv

from large_legislative_models.environment import meltingpot_utils

"from: https://github.com/Farama-Foundation/SuperSuit/blob/fda0bb0de597b8f86fb5fe8f33e7c18f987fb006/supersuit/vector/vector_constructors.py#L10"

PLAYER_STR_FORMAT = "player_{index}"


def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


def create_parallel_games(vec_env, num_vec_envs, multiprocessing):
    if num_vec_envs == 1:
        vec_env = OneParallelGameWrapper(vec_env)
    else:
        if multiprocessing:
            vec_env = AsyncVectorEnv(*vec_env_args(vec_env, num_vec_envs))
        else:
            vec_env = SyncVectorEnv(*vec_env_args(vec_env, num_vec_envs))
    return vec_env


"FROM: https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/markov_vector_wrapper.py"


class MarkovVectorEnv(gymnasium.vector.VectorEnv):
    def __init__(self, par_env, black_death=False):
        """
        parameters:
            - par_env: the pettingzoo Parallel environment that will be converted to a gymnasium vector environment
            - black_death: whether to give zero valued observations and 0 rewards when an agent is done, allowing for environments with multiple numbers of agents.
                            Is equivalent to adding the black death wrapper, but somewhat more efficient.

        The resulting object will be a valid vector environment that has a num_envs
        parameter equal to the max number of agents, will return an array of observations,
        rewards, dones, etc, and will reset environment automatically when it finishes
        """
        self.par_env = par_env
        self.metadata = par_env.metadata
        self.render_mode = par_env.unwrapped.render_mode
        self.single_observation_space = par_env.observation_space(par_env.possible_agents[0])["RGB"]
        self.action_space = par_env.action_space(par_env.possible_agents[0])
        self.num_envs = len(par_env.possible_agents)
        self.observation_space = Box(
            low=self.single_observation_space.low[0][0][0],
            high=self.single_observation_space.high[0][0][0],
            shape=(self.num_envs,) + self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )

        self.black_death = black_death

    def concat_obs(self, obs_dict):
        obs_list = []
        for i, agent in enumerate(self.par_env.possible_agents):
            if agent not in obs_dict:
                raise AssertionError(
                    "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
                )
            obs_list.append(obs_dict[agent]["RGB"])

        return concatenate(
            self.single_observation_space,
            obs_list,
            create_empty_array(self.single_observation_space, self.num_envs),
        )

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def reset(self, seed=None, options=None):
        _observations, infos = self.par_env.reset(seed=seed, options=options)
        observations = self.concat_obs(_observations)
        return observations, infos["player_0"]

    def apply_principal_action(self, action):
        self.par_env.apply_principal_action(action)

    def step(self, actions):
        actions = list(iterate(self.action_space, actions))
        agent_set = set(self.par_env.agents)
        act_dict = {agent: actions[i] for i, agent in enumerate(self.par_env.possible_agents) if agent in agent_set}
        observations, rewards, terms, truncs, infos = self.par_env.step(act_dict)

        rews = torch.stack([rewards.get(agent) for agent in self.par_env.possible_agents])
        tms = np.array(
            [terms.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )
        tcs = np.array(
            [truncs.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )

        observations = self.concat_obs(observations)

        return observations, rews, tms, tcs, infos["player_0"]

    def render(self):
        return self.par_env.render()

    def close(self):
        return self.par_env.close()

    def env_is_wrapped(self, wrapper_class):
        """
        env_is_wrapped only suppors vector and gymnasium environments
        currently, not pettingzoo environments
        """
        return [False] * self.num_envs


class OneParallelGameWrapper():
    """ If we only have one parallel game, no need for parallelised envs.
    However, code expects lists and multi-dimensional tensors, so need an adapter. """
    def __init__(self, env):

        self.env = env
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        self.single_observation_space = env.single_observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset()
        return torch.from_numpy(observations), [infos]

    def apply_principal_action(self, action):
        self.env.apply_principal_action(action[0])

    def step(self, actions):
        observations, rews, tms, tcs, infos = self.env.step(actions)
        return torch.from_numpy(observations), rews, torch.from_numpy(tms), torch.from_numpy(tcs), [infos]

    def close(self):
        return self.env.close()


@iterate.register(Discrete)
def iterate_discrete(space, items):
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f"Unable to iterate over the following elements: {items}")


class _MeltingPotPettingZooEnv(pettingzoo_utils.ParallelEnv):
    """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""

    def __init__(self, env_config, max_cycles):
        self.env_config = config_dict.ConfigDict(env_config)
        self.max_cycles = max_cycles
        self._env = substrate.build_from_config(self.env_config, roles=self.env_config.default_player_roles)
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)]
        observation_space = meltingpot_utils.remove_world_observations_from_space(
            meltingpot_utils.spec_to_space(self._env.observation_spec()[0])
        )
        self.observation_space = functools.lru_cache(maxsize=None)(lambda agent_id: observation_space)
        action_space = meltingpot_utils.spec_to_space(self._env.action_spec()[0])
        self.action_space = functools.lru_cache(maxsize=None)(lambda agent_id: action_space)
        self.state_space = meltingpot_utils.spec_to_space(self._env.observation_spec()[0]["WORLD.RGB"])

    def state(self):
        return self._env.observation()

    def reset(self, seed=None, options=None):
        """See base class."""
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        observations, world_obs = meltingpot_utils.timestep_to_observations(timestep)
        infos = {agent: {} for agent in self.agents}
        infos[self.agents[0]]["world_obs"] = world_obs
        return observations, infos

    def step(self, action):
        """See base class."""
        actions = [action[agent] for agent in self.agents]
        timestep = self._env.step(actions)
        rewards = {agent: timestep.reward[index] for index, agent in enumerate(self.agents)}
        self.num_cycles += 1
        done = timestep.last() or self.num_cycles >= self.max_cycles
        dones = {agent: done for agent in self.agents}

        observations, world_obs = meltingpot_utils.timestep_to_observations(timestep)
        infos = {agent: {} for agent in self.agents}
        infos[self.agents[0]]["world_obs"] = world_obs
        if done:
            self.agents = []
        return observations, rewards, dones, dones, infos

    def close(self):
        """See base class."""
        self._env.close()

    def render(self, mode="human", filename=None):
        rgb_arr = self.state()["WORLD.RGB"]
        if mode == "human":
            plt.cla()
            plt.imshow(rgb_arr, interpolation="nearest")
            if filename is None:
                plt.show(block=False)
            else:
                plt.savefig(filename)
            return None
        return rgb_arr


class ParallelEnv(_MeltingPotPettingZooEnv, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config, max_steps):
        gym_utils.EzPickle.__init__(self, env_config, max_steps)
        _MeltingPotPettingZooEnv.__init__(self, env_config, max_steps)


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset(seed=42)
        (array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32), {})
    """

    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=sum(env.num_envs for env in self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(self.single_observation_space, n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def apply_principal_action(self, actions):
        for i, env in enumerate(self.envs):
            env.apply_principal_action(actions[i])

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations = []
        infos = []
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos.append(info)

        self.observations = torch.from_numpy(np.concatenate(observations))

        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)

    def step_wait(self) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, np.array_split(list(self._actions), len(self.envs)))):
            (
                observation,
                self._rewards[i * self.envs[0].num_envs : (i + 1) * self.envs[0].num_envs],
                self._terminateds[i * self.envs[0].num_envs : (i + 1) * self.envs[0].num_envs],
                self._truncateds[i * self.envs[0].num_envs : (i + 1) * self.envs[0].num_envs],
                info,
            ) = env.step(action)

            observations.append(observation)
            infos.append(info)

        self.observations = torch.from_numpy(np.concatenate(observations))

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            torch.from_numpy(np.copy(self._rewards)),
            torch.from_numpy(np.copy(self._terminateds)),
            torch.from_numpy(np.copy(self._truncateds)),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel.

    It uses ``multiprocessing`` processes, and pipes for communication.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.AsyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset(seed=42)
        (array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32), {})
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        shared_memory: bool = True,
        copy: bool = True,
        context: Optional[str] = None,
        daemon: bool = True,
        worker: Optional[Callable] = None,
    ):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through
                shared variables. This can improve the efficiency if the observations are large (e.g. images).
            copy: If ``True``, then the :meth:`~AsyncVectorEnv.reset` and :meth:`~AsyncVectorEnv.step` methods
                return a copy of the observations.
            context: Context for `multiprocessing`_. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if
                the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children,
                so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one.
                Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.

        Warnings:
            worker is an advanced mode option. It provides a high degree of flexibility and a high chance
            to shoot yourself in the foot; thus, if you are writing your own worker, it is recommended to start
            from the code for ``_worker`` (or ``_worker_shared_memory``) method, and add changes.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
            ValueError: If observation_space is a custom space (i.e. not a default space in Gym,
                such as gymnasium.spaces.Box, gymnasium.spaces.Discrete, or gymnasium.spaces.Dict) and shared_memory is True.
        """
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        dummy_env = env_fns[0]()
        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()
        del dummy_env
        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(self.single_observation_space, n=self.num_envs, ctx=ctx)
                self.observations = np.frombuffer(
                    _obs_buffer.get_obj(), dtype=self.single_observation_space.dtype
                ).reshape((self.num_envs * self.single_observation_space.shape[0],) + self.single_observation_space.shape[1:])
            except CustomSpaceError as e:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gymnasium observation spaces "
                    "(i.e. custom spaces inheriting from `gymnasium.Space`), and is "
                    "only compatible with default Gymnasium spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                ) from e
        else:
            _obs_buffer = None
            self.observations = create_empty_array(self.single_observation_space, n=self.num_envs, fn=np.zeros)

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()

        """ Only shared memory worker has been modified. """
        target = _worker_shared_memory
        target = worker or target
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()

    def apply_principal_action(self, actions):
        for i, env_fn in enumerate(self.env_fns):
            env_fn().apply_principal_action(actions[i])

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_wait`.

        Args:
            seed: List of seeds for each environment
            options: The reset option

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`step_async`). This can be caused by two consecutive
                calls to :meth:`reset_async`, with no call to :meth:`reset_wait` in between.
        """
        self._assert_is_running()

        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                self._state.value,
            )

        for pipe, single_seed in zip(self.parent_pipes, seed):
            single_kwargs = {}
            if single_seed is not None:
                single_kwargs["seed"] = single_seed
            if options is not None:
                single_kwargs["options"] = options

            pipe.send(("reset", single_kwargs))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            timeout: Number of seconds before the call to `reset_wait` times out. If `None`, the call to `reset_wait` never times out.
            seed: ignored
            options: ignored

        Returns:
            A tuple of batched observations and list of dictionaries

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`reset_wait` was called without any prior call to :meth:`reset_async`.
            TimeoutError: If :meth:`reset_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(f"The call to `reset_wait` has timed out after {timeout} second(s).")

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        infos = []
        results, info_data = zip(*results)
        for info in info_data:
            infos.append(info)

        if not self.shared_memory:
            self.observations = torch.from_numpy(np.concatenate(results))

        return (torch.from_numpy(deepcopy(self.observations)) if self.copy else torch.from_numpy(self.observations)), infos

    def step_async(self, actions: np.ndarray):
        """Send the calls to :obj:`step` to each sub-environment.

        Args:
            actions: Batch of actions. element of :attr:`~VectorEnv.action_space`

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`reset_async`). This can be caused by two consecutive
                calls to :meth:`step_async`, with no call to :meth:`step_wait` in
                between.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        actions = np.array_split(actions, len(self.parent_pipes))
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(
        self, timeout: Optional[Union[int, float]] = None
    ) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out. If ``None``, the call to :meth:`step_wait` never times out.

        Returns:
             The batched environment step information, (obs, reward, terminated, truncated, info)

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`step_wait` was called without any prior call to :meth:`step_async`.
            TimeoutError: If :meth:`step_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(f"The call to `step_wait` has timed out after {timeout} second(s).")

        observations_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        successes = []
        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            successes.append(success)
            if success:
                obs, rew, terminated, truncated, info = result

                observations_list.append(obs)
                rewards.append(rew)
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos.append(info)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = torch.from_numpy(np.concatenate(observations_list))

        return (
            torch.from_numpy(deepcopy(self.observations)) if self.copy else torch.from_numpy(self.observations),
            torch.concatenate(rewards),
            torch.from_numpy(np.concatenate(terminateds)),
            torch.from_numpy(np.concatenate(truncateds)),
            infos,
        )

    def apply_principal_action(self, principal_action):
        """ Apply principal actions in all sub-environments. """

        self._assert_is_running()

        for pipe, action in zip(self.parent_pipes, principal_action):
            pipe.send(("apply_principal_action", action))

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting " f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `step_wait` times out.
                If `None` (default), the call to `step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_wait` without any prior call to `call_async`.
            TimeoutError: The call to `call_wait` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(f"The call to `call_wait` has timed out after {timeout} second(s).")

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling `set_attr` while waiting for a pending call to complete.
        """
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting " f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def close_extras(self, timeout: Optional[Union[int, float]] = None, terminate: bool = False):
        """Close the environments & clean up the extra resources (processes and pipes).

        Args:
            timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated.
            terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated.

        Raises:
            TimeoutError: If :meth:`close` timed out.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(f"Calling `close` while waiting for a pending call to `{self._state.value}` to complete.")
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _check_spaces(self):
        self._assert_is_running()
        spaces = (self.single_observation_space, self.single_action_space)
        for pipe in self.parent_pipes:
            pipe.send(("_check_spaces", spaces))
        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        same_observation_spaces, same_action_spaces = zip(*results)
        if not all(same_observation_spaces):
            raise RuntimeError(
                "Some environments have an observation space different from "
                f"`{self.single_observation_space}`. In order to batch observations, "
                "the observation spaces from all environments must be equal."
            )
        if not all(same_action_spaces):
            raise RuntimeError(
                "Some environments have an action space different from "
                f"`{self.single_action_space}`. In order to batch actions, the "
                "action spaces from all environments must be equal."
            )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(f"Trying to operate on `{type(self).__name__}`, after a call to `close()`.")

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(f"Received the following error from Worker-{index}: {exctype.__name__}: {value}")
            logger.error(f"Shutting down Worker-{index}.")
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error("Raising the last exception back to the main process.")
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, info), True))
            elif command == "apply_principal_action":
                env.apply_principal_action(data)
                pipe.send((None, True))
            elif command == "step":
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with " f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(((data[0] == observation_space, data[1] == env.action_space), True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
