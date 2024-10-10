from abc import ABC, abstractmethod
import torch
from tensordict import TensorDict
from torch import zeros
from large_legislative_models.utils import welfare


class FixedLengthTrajectory:
    def __init__(self, trajectory_length, base_shape, obs_shape, action_shape, device):
        self.shared_shape = (trajectory_length, base_shape)
        self.device = device
        self.tensordict = TensorDict(
            {
                "logprobs": zeros(self.shared_shape),
                "rewards": zeros(self.shared_shape),
                "dones": zeros(self.shared_shape),
                "values": zeros(self.shared_shape),
                "entropy": zeros(self.shared_shape),
                "actions": zeros(self.shared_shape + action_shape),
                "obs": zeros(self.shared_shape + obs_shape),
            },
            batch_size=self.shared_shape,
        )
        self.flattened_dict = None

    def record_step(self, step_data: TensorDict, trajectory_step):
        self.tensordict[trajectory_step] = step_data

    def _flatten(self):
        self.flattened_dict = self.tensordict.flatten(end_dim=1)

    def get_batch(self, indices):
        if self.flattened_dict is None:
            self._flatten()
        # if more stuff that isn't used for optimisation is added to this buffer, consider sending to GPU selectively.
        return self.flattened_dict[indices].to(self.device)

    def reset(self):
        self.flattened_dict = None
        """ Continually overwriting the rewards tensor still flows gradient
        back to old rewards. This gradient needs to be cut before we overwrite.
        For example: a[0] = x, a[0] = y, a.sum().backward() -> x.grad populated!
        """
        self.tensordict["rewards"].detach_()


class EpisodeBuffer(ABC):
    """ Base class for training and validation episodes.
    The primary difference between the two is that validation episodes
    do not keep a sampling_horizon fixed-length buffer, which has large
    memory overhead.
    """

    def __init__(
        self,
        envs,  # needed as buffers are responsible for resetting environment
        num_parallel_games,
        episode_length,
        tax_rates_for_this_ep,
        record_world_obs,
    ):
        self.num_agents = envs.num_agents
        self.num_parallel_games = num_parallel_games
        self.record_world_obs = record_world_obs

        """ Tensor[num_parallel_games, num_tax_brackets], holding tax brackets of each game.
        Needed in episode buffers as they are responsible for adding tax rate indicators to observations. """
        self.tax_rates_for_this_ep = tax_rates_for_this_ep

        """ Agent cumulative reward with no gradient, for logging purposes. """
        self.logging_agent_cumulative_rewards = zeros(num_parallel_games * self.num_agents)

        self.principal_cumulative_reward = zeros(num_parallel_games)
        self.principal_reward_trajectory = zeros(episode_length, num_parallel_games)

        """ Environment-dependent storage. Shouldn't be accessed directly. """
        self._env_storage = EpisodeEnvironmentStorage(envs.name, num_parallel_games, self.num_agents, episode_length)

        """ Reset the environment and store the first "next" observation and done. """
        reset_obs, reset_infos = envs.reset()
        self.cache_next(
            agent_next_done=zeros(num_parallel_games * self.num_agents),
            agent_next_obs=reset_obs.float(),
        )

        """ First frame of video comes from reset - these come as numpy arrays. Needs to come after env reset."""
        self.first_parallel_game_world_obs = [reset_infos[0]["world_obs"]]

    @abstractmethod
    def record(self, *args, **kwargs):
        """Record one timestep of data to buffers."""
        pass

    @abstractmethod
    def cache_next(self, *args, **kwargs):
        """Cache data needed at next timestep loop iteration."""
        pass

    @abstractmethod
    def get_cached(self, *args, **kwargs):
        """Retrieve data cached at previous timestep loop iteration."""
        pass

    @abstractmethod
    def reset_agent_trajectory(self, *args, **kwargs):
        """Reset gradients on the continually overwritten agent trajectory."""
        pass

    @abstractmethod
    def record(self, *args, **kwargs):
        pass

    def get_formatted_infos(self, prefix):
        return self._env_storage.get_formatted_infos(prefix)

    def _common_recording(self, agent_step_data: TensorDict, info, episode_step):
        """ Recording common to training and validation episodes. """
        
        if self.record_world_obs:
            self.first_parallel_game_world_obs.append(info[0]["world_obs"])

        self.logging_agent_cumulative_rewards += agent_step_data["rewards"].detach()

        """ Record principal reward trajectory, detaching gradients. """
        last_principal_cumulative_reward = self._env_storage.get_principal_cumulative_reward().detach()
        self._env_storage.record_step(agent_step_data, info, episode_step)
        self.principal_cumulative_reward = self._env_storage.get_principal_cumulative_reward().detach()
        self.principal_reward_trajectory[episode_step] = self.principal_cumulative_reward - last_principal_cumulative_reward

    def prepare_observation(self, observation):
        """Operations that need to be done on observations between what the envs output and what
        the learning procedures need.

        Args:
            observation (Tensor): observation from the environments

        Returns:
            (Tensor): observation ready for use in learning
        """

        """
        Observation pixel values are in [0,255], so we scale all transformations
        applied to observations up to this point to lie in this range too.
        This is done in-place for memory considerations at the cost of safety and clarity.
        """
        observation /= 255.0
        return observation


class TrainingEpisode(EpisodeBuffer):

    def __init__(
        self,
        agent_trajectory: FixedLengthTrajectory,
        envs,
        num_parallel_games,
        episode_length,
        tax_rates_for_this_ep,
        record_world_obs,
    ):
        """Trajectory of length sampling_horizon that is overwritten as we chunk through the episode.
        Validation episodes do not need this as they only collect episode-length data.
        Needed in parent initialisation for caching reset observation.
        """
        self.agent_trajectory = agent_trajectory

        super().__init__(
            envs=envs,
            num_parallel_games=num_parallel_games,
            episode_length=episode_length,
            tax_rates_for_this_ep=tax_rates_for_this_ep,
            record_world_obs=record_world_obs,
        )

    def record(self, agent_step_data: TensorDict, info, trajectory_step, episode_step):
        """Record one timestep of data to buffers."""

        """ Step recording common to all episode buffers. """
        self._common_recording(agent_step_data, info, episode_step)

        """ Record this step's data in agent trajectory buffer. """
        self.agent_trajectory.record_step(agent_step_data, trajectory_step)

    def cache_next(self, agent_next_done, agent_next_obs):
        """Next observation and done are cached in the agent trajectory buffer."""
        self.agent_trajectory.next_done = agent_next_done
        self.agent_trajectory.next_obs = self.prepare_observation(agent_next_obs)

    def get_cached(self):
        return self.agent_trajectory.next_obs, self.agent_trajectory.next_done

    def reset_agent_trajectory(self):
        self.agent_trajectory.reset()


class ValidationEpisode(EpisodeBuffer):

    def __init__(self, envs, num_parallel_games, episode_length, tax_rates_for_this_ep, record_world_obs):
        super().__init__(
            envs=envs,
            num_parallel_games=num_parallel_games,
            episode_length=episode_length,
            tax_rates_for_this_ep=tax_rates_for_this_ep,
            record_world_obs=record_world_obs,
        )

        self.agent_episode_log_probs = zeros(episode_length, num_parallel_games, self.num_agents)

    def record(self, agent_step_data: TensorDict, info, trajectory_step, episode_step):

        """ Step recording common to all episode buffers. """
        self._common_recording(agent_step_data, info, episode_step)

        """Record differentiable agent logprobs for validation trajectory.
        Careful not to use this step data elsewhere forgetting log_probs have gradient! """
        self.agent_episode_log_probs[episode_step] = (
            agent_step_data["logprobs"].clone().view(-1, self.num_agents)
        )  # STILL HAS GRADIENT

    def cache_next(self, agent_next_done, agent_next_obs):
        """Since validation episodes do not have an agent trajectory buffer,
        next observation and done are cached in the episode buffer itself."""
        self.agent_next_done = agent_next_done
        self.agent_next_obs = self.prepare_observation(agent_next_obs)

    def get_cached(self):
        return self.agent_next_obs, self.agent_next_done

    def reset_agent_trajectory(self):
        pass

    def get_episode_principal_observation(self):
        """ Validation episodes collect certain pieces of information that principals
        are allowed to use as observations to inform their next decision.
        This is returned from here, providing an information barrier controlling what
        principals can and cannot observe. """
        return self._env_storage.get_episode_principal_observation()


class EpisodeEnvironmentStorage:
    """ Handles environment specific storage. All gritty details and conditionals should
    ideally happen here only. """

    def __init__(self, env_name, num_parallel_games, num_agents, episode_length) -> None:
        assert env_name in ["commons_harvest__open", "clean_up", "cer"]
        self.env_name = env_name
        self.num_agents = num_agents
        self.num_parallel_games = num_parallel_games
        self.episode_step = 0

        """ Agent cumulative reward for principal reward calculation. """
        self.agent_cumulative_rewards = zeros(num_parallel_games * self.num_agents)

        match self.env_name:
            case "commons_harvest__open":
                self.current_num_apples_per_game = zeros(num_parallel_games)
                self.apples_trajectory = zeros(episode_length, num_parallel_games)
                self.cumulative_raw_rewards = zeros(num_parallel_games, num_agents)
                self.regrowth_trajectory = zeros(episode_length, num_parallel_games)
            case "clean_up":
                self.current_num_apples_per_game = zeros(num_parallel_games)
                self.total_incentive_given_per_game = zeros(num_parallel_games)
                self.total_num_cleaned_per_game = zeros(num_parallel_games)
                self.cumulative_raw_rewards = zeros(num_parallel_games, num_agents)
                self.regrowth_trajectory = zeros(episode_length, num_parallel_games)
                self.num_cleaned_trajectory = zeros(episode_length, num_parallel_games)
            case "cer":
                self.num_lever0_per_step = zeros(num_parallel_games)
                self.num_lever1_per_step = zeros(num_parallel_games)
                self.num_lever2_per_step = zeros(num_parallel_games)
                self.num_door_per_step = zeros(num_parallel_games)
                self.num_start_per_step = zeros(num_parallel_games)
                self.indicator = zeros(num_parallel_games)
                self.door_status = zeros(num_parallel_games)
                self.cumulative_raw_total_reward = zeros(num_parallel_games)

    def record_step(self, agent_step_data: TensorDict, info, episode_step):
        self.agent_cumulative_rewards += agent_step_data["rewards"].detach()
        self.episode_step += 1

        match self.env_name:
            case "commons_harvest__open":
                self.current_num_apples_per_game = torch.Tensor([
                    game_info["current_num_apples"] for game_info in info
                ])
                self.cumulative_raw_rewards += torch.stack([
                    game_info["raw_rewards"] for game_info in info
                ])
                self.apples_trajectory[episode_step] = self.current_num_apples_per_game
            case "clean_up":
                self.current_num_apples_per_game = torch.Tensor([
                    game_info["current_num_apples"] for game_info in info
                ])
                self.total_incentive_given_per_game += torch.tensor([
                    game_info["total_incentive_given"] for game_info in info
                ])
                self.total_num_cleaned_per_game += torch.tensor([
                    game_info["num_cleaned"] for game_info in info
                ])
                self.cumulative_raw_rewards += torch.stack([
                    game_info["raw_rewards"] for game_info in info
                ])
                self.regrowth_trajectory[episode_step] = torch.Tensor([
                    game_info['apples_regrown'] for game_info in info
                ])
                self.num_cleaned_trajectory[episode_step] = torch.Tensor([
                    game_info['num_cleaned'] for game_info in info
                ])
            case "cer":
                self.num_lever0_per_step += torch.Tensor([
                    game_info["num_lever_0"] for game_info in info
                ])
                self.num_lever1_per_step += torch.Tensor([
                    game_info["num_lever_1"] for game_info in info
                ])
                self.num_lever2_per_step += torch.Tensor([
                    game_info["num_lever_2"] for game_info in info
                ])
                self.num_door_per_step += torch.Tensor([
                    game_info["num_door"] for game_info in info
                ])
                self.num_start_per_step += torch.Tensor([
                    game_info["num_start"] for game_info in info
                ])
                self.indicator += torch.tensor([
                    game_info["indicator"] for game_info in info
                ])
                self.door_status += torch.tensor([
                    game_info["door_open"] for game_info in info
                ])
                self.cumulative_raw_total_reward += torch.tensor([
                    game_info["total_raw_reward"] for game_info in info
                ])

    def get_episode_principal_observation(self):
        """ Returns principal observation for this episode.
        All principals should have access to the same observation, and then
        can choose to do whatever they want with it.
        Should only be called after an episode has finished. """

        match self.env_name:
            case "commons_harvest__open":
                principal_observation = {
                    "apples_trajectory": self.apples_trajectory,
                    "cumulative_agent_raw_rewards": self.cumulative_raw_rewards,
                    "regrowth_trajectory": self.regrowth_trajectory,    
                }
            case "clean_up":
                principal_observation = {
                    "cumulative_agent_raw_rewards": self.cumulative_raw_rewards,
                    "num_cleaned": self.total_num_cleaned_per_game,
                    "regrowth_trajectory": self.regrowth_trajectory,
                    "num_cleaned_trajectory": self.num_cleaned_trajectory,
                }
            case "cer":
                principal_observation = {
                    "num_lever0_per_step": self.num_lever0_per_step/self.episode_step,
                    "num_lever1_per_step": self.num_lever1_per_step/self.episode_step,
                    "num_lever2_per_step": self.num_lever2_per_step/self.episode_step,
                    "num_door_per_step": self.num_door_per_step/self.episode_step,
                    "num_start_per_step": self.num_start_per_step/self.episode_step,
                    "indicator": self.indicator/self.episode_step,
                    "door_open": self.door_status/self.episode_step,
                }
        return principal_observation

    def get_principal_cumulative_reward(self):
        """ Returns principal's cumulative reward. """

        match self.env_name:
            case "commons_harvest__open":
                principal_cumulative_reward = welfare(self.agent_cumulative_rewards.view(-1, self.num_agents))
            case "clean_up":
                raw_reward_welfare = welfare(self.cumulative_raw_rewards)
                principal_cumulative_reward = raw_reward_welfare
            case "cer":
                principal_cumulative_reward = self.cumulative_raw_total_reward
        return principal_cumulative_reward
    
    def get_formatted_infos(self, prefix):
        """ Returns infos formatted for logging. """

        result = {}

        match self.env_name:
            case "commons_harvest__open":
                for i in range(self.num_parallel_games):
                    result[f"{prefix}game {i} apples remaining"] = self.current_num_apples_per_game[i]
            
            case "clean_up":
                for i in range(self.num_parallel_games):
                    result[f"{prefix}game {i} apples remaining"] = self.current_num_apples_per_game[i]
                    result[f"{prefix}game {i} total_incentive_given"] = self.total_incentive_given_per_game[i]
                    result[f"{prefix}game {i} num_cleaned"] = self.total_num_cleaned_per_game[i]
                    result[f"{prefix}game {i} total_num_apples_collected"] = (
                        self.cumulative_raw_rewards.sum(dim=1)[i]
                        )
            case "cer":
                for i in range(self.num_parallel_games):
                    result[f"{prefix}game {i} num_lever0_per_step"] = self.num_lever0_per_step[i].item()/self.episode_step
                    result[f"{prefix}game {i} num_lever1_per_step"] = self.num_lever1_per_step[i].item()/self.episode_step
                    result[f"{prefix}game {i} num_lever2_per_step"] = self.num_lever2_per_step[i].item()/self.episode_step
                    result[f"{prefix}game {i} num_door_per_step"] = self.num_door_per_step[i].item()/self.episode_step
                    result[f"{prefix}game {i} num_start_per_step"] = self.num_start_per_step[i].item()/self.episode_step
                    result[f"{prefix}game {i} indicator"] = self.indicator[i].item()/self.episode_step
                    result[f"{prefix}game {i} door_open"] = self.door_status[i].item()/self.episode_step
                    result[f"{prefix}game {i} total_raw_reward_per_step"] = (
                        self.cumulative_raw_total_reward[i]
                    )/self.episode_step
        
        return result
