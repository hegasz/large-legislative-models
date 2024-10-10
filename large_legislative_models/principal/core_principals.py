import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from large_legislative_models.principal import Principal
from large_legislative_models.principal.llm_utils import api_handler_factory, format_demonstration, retry_with_prompt_adjustment
from large_legislative_models.principal.prompts import get_prompt_backbone
from large_legislative_models.training.optimize import step_principal_agent_nets
from large_legislative_models.neural.agent_architectures import DesignerNet, PrincipalAgent
from large_legislative_models.utils.logger import logger


class Designer(Principal):
    def __init__(self, args, num_agents, device, principal_obs_length, num_brackets):
        super().__init__(args)
        assert (
            args.freeze_agent_net_core
        ), "Meta-gradients on full net are highly memory-intensive. Only actor and critic heads trained."
        assert (
            args.num_convergence == 1 and args.num_validation == 1
        ), "This method is only defined for one convergence and validation episode."
        assert (
            args.num_parallel_games == 1
        ), "Asynchronous parallel games will require autograd crossing process boundaries. Best of luck. "

        self.device = device
        self.num_agents = num_agents
        self.principal_obs_length = principal_obs_length

        """ Incentive function net. """
        self.incentive_function = DesignerNet(
            principal_obs_length=principal_obs_length,
            h_dim=args.aid_hidden_dim,
            num_brackets=num_brackets,
            output_multiplier=args.upper_bound
        ).to(device)

        """ Optimizer for incentive / tax function net. """
        self.optimizer = optim.Adam(
            self.incentive_function.parameters(),
            lr=args.principal_lr,
            eps=args.adam_eps,
        )

        """ Initialise running mean baseline. """
        self.mean_rewards = torch.zeros(args.num_parallel_games)
        self.num_mean_updates = 1

        """ Previous episode rewards we feed as input to incentive / tax function - set to None for initialisation. """
        self.cached_principal_observation = None

    def _process_observation(self, principal_observation):
        """ Process a single principal historical observation into an input to network.

        Args:
            principal_observation (dict): set of allowed observations from single validation episode.

        Returns:
            (Tensor): historical observation processed for input to network.
        """

        if self.args.principal_gets_historical and principal_observation is not None:
            match self.args.env_name:
                case "cer":
                    """ Historical observation is a combination of a variety of agent
                    movement / position trackers and the door's status. """
                    obs = torch.stack([
                        principal_observation["num_lever0_per_step"],
                        principal_observation["num_lever1_per_step"],
                        principal_observation["num_lever2_per_step"],
                        principal_observation["num_door_per_step"],
                        principal_observation["num_start_per_step"],
                        principal_observation["door_open"],
                    ]).T
                case "commons_harvest__open":
                    """ Historical observation is a full trajectory of the number of apples at each step. """
                    obs = principal_observation["apples_trajectory"].T/64
                case "clean_up":
                    """ Historical observation is a concatenation of apple regrowth and cleaning action
                    trajectories, both downsampled by a factor of two. """
                    regrowth_traj = principal_observation["regrowth_trajectory"].T
                    num_cleaned_traj = principal_observation["num_cleaned_trajectory"].T
                    downsampled_regrowth = regrowth_traj[:, :: 2]
                    downsampled_num_cleaned = num_cleaned_traj[:, :: 2]
                    obs = torch.cat([downsampled_regrowth, downsampled_num_cleaned], dim=-1)
        else:
            match self.args.env_name:
                case "cer":
                    obs = torch.full((self.args.num_parallel_games, self.principal_obs_length-3), 1).float()
                case "commons_harvest__open":
                    obs = torch.full((self.args.num_parallel_games, self.principal_obs_length), 1).float()
                case "clean_up":
                    obs = torch.full((self.args.num_parallel_games, self.principal_obs_length), 1).float()

        return obs
    
    def set_tax_vals(self, indicator, principal_step):
        """ Decide and set tax values / incentives. 

        Args:
            indicator (Tensor): one-hot encoding of environment indicator (e.g. lever activated
                                for CER, and empty if not applicable so concatenation adds nothing)
            principal_step (int): current principal step number

        Returns:
            (Tensor): tax rates / incentives chosen - gradient flows through these
        """        

        """ Extract historical observation to act on from information principal is allowed to observe. """
        historical_obs = self._process_observation(self.cached_principal_observation)
        
        """ Save observation as a field as a one-step buffer for optimisation later. """
        observation_to_act_on = torch.cat([indicator, historical_obs], dim=1).to(self.device)

        """ Decide next tax values. """
        tax_vals_per_game = self.incentive_function(observation_to_act_on).cpu()

        """ Return tax values with gradient flow. """
        return tax_vals_per_game

    def update(self, ctx, principal_step, episode_buffers):
        """ Update principal parameters using validation episodes.

        Args:
            ctx (Context): training context
            principal_step (int): principal step number
            episode_buffers (list[ValidationEpisode]): list of validation episode buffers
        """

        """ Method only uses one validation episode. """
        validation_episode = episode_buffers[0]

        """ Record information from validation episode that principal is allowed access to. """
        self.cached_principal_observation = validation_episode.get_episode_principal_observation()

        """ Principal reward-to-go in validation trajectory. """
        ones = torch.ones_like(validation_episode.principal_reward_trajectory)
        # gamma_prod is $$[\gamma, \gamma^2, \dots, \gamma^n]$$
        gamma_prod = torch.cumprod(ones * self.args.gamma, dim=0)
        # principal_returns here is $$\left[\sum_{t=1}^n{r_t\gamma^t}, \sum_{t=2}^n{r_t\gamma^t}, \dots, r_n\gamma^n\right]$$
        principal_returns = torch.flip(
            torch.cumsum(
                torch.flip(validation_episode.principal_reward_trajectory * gamma_prod, dims=[0]),
                dim=0,
            ),
            dims=[0],
        )
        # principal_returns here is $$\left[\left(r_1+r_2\gamma+\dots\right),\left(r_2+r_3\gamma+\dots\right),\dots, r_n\right]$$
        principal_returns = principal_returns / gamma_prod
        baselined_returns = principal_returns - self.mean_rewards


        """ Update running mean value estimator. (i.e. "step critic estimator")"""
        principal_rewards_per_game = validation_episode.principal_reward_trajectory.sum(dim=0)
        self.mean_rewards += (principal_rewards_per_game - self.mean_rewards) / self.num_mean_updates
        self.num_mean_updates += 1

        """ Principal policy gradient loss.
        Agent policy parameters have been updated in training episode using rewards dependent on tax function parameters.
        Since these new policies were update differentiably, their parameters have gradient dependency on tax function parameters.
        In validation episode, we collected log-probabilities of actions produced by these updated nets.
        These log-probabilities retain gradient dependency on tax function parameters.
        We use them in a policy gradient loss with the principal's validation episode reward trajectory to step tax function parameters.
        """
        # Action log-probabilities in validation episode for updated agent policies, summed over agents in each parallel game.
        sum_agents_log_probs = validation_episode.agent_episode_log_probs.sum(dim=-1)
        # Policy gradient loss to differentiate back tax function's effect on principal returns in validation episode.
        principal_pg_loss = -(sum_agents_log_probs * baselined_returns).sum()

        """ Step tax function """
        principal_pg_loss.backward()
        nn.utils.clip_grad_norm_(self.incentive_function.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_params(self, principal_step):
        torch.save(
            self.incentive_function.state_dict(),
            f"./saved_params/step{principal_step}/seed_{self.args.seed}_designer_net.pt",
        )


class DualRLPrincipal(Principal):

    def __init__(self, args, num_agents, device, principal_obs_length, num_brackets):
        super().__init__(args)
        self.device = device
        self.num_agents = num_agents
        self.principal_obs_length = principal_obs_length

        """ Principal agent net. """        
        self.principal_agent = PrincipalAgent(
            principal_obs_length=principal_obs_length,
            num_brackets=num_brackets,
            discretized_bracket_length=len(self.args.discretization_set),
            hidden_dim=args.dual_rl_hidden_dim
        ).to(device)        

        """ Optimizer for principal agent net. """
        self.optimizer = optim.Adam(
            self.principal_agent.parameters(),
            lr=self.args.principal_lr,
            eps=self.args.adam_eps,
        )

        """ First cached observation from previous episode is None. """
        self.cached_principal_obs_list = None

        """ Since this principal can produce NO-OP "do not change tax rate" actions, we must initialise some first tax rate for NO-OP masking. """
        self.previous_proposed_tax_rates = torch.zeros(self.args.num_parallel_games, num_brackets)

    def _process_observation(self, principal_obs_list):
        """ Process a list of principal historical observations - one from each validation episode.

        Args:
            principal_observation (list[dict]): list of sets of allowed observations from all validation episodes.

        Returns:
            (Tensor): historical observation averaged over validation episodes and processed for input to network.
        """

        if self.args.principal_gets_historical and principal_obs_list is not None:
            match self.args.env_name:
                case "cer":
                    """ Historical observation is a combination of a variety of agent
                    movement / position trackers and the door's status. """
                    obs = torch.zeros(self.args.num_parallel_games, self.principal_obs_length-3)
                    for i in range(len(principal_obs_list)):
                        obs += torch.stack([
                            principal_obs_list[i]["num_lever0_per_step"],
                            principal_obs_list[i]["num_lever1_per_step"],
                            principal_obs_list[i]["num_lever2_per_step"],
                            principal_obs_list[i]["num_door_per_step"],
                            principal_obs_list[i]["num_start_per_step"],
                            principal_obs_list[i]["door_open"],
                        ]).T
                    obs /= len(principal_obs_list)
                case "commons_harvest__open":
                    """ Historical observation is a full trajectory of the number of apples at each step. """
                    obs = torch.zeros(self.args.num_parallel_games, self.principal_obs_length)
                    for i in range(len(principal_obs_list)):
                        obs += principal_obs_list[i]["apples_trajectory"].T/64
                    obs /= len(principal_obs_list)
                case "clean_up":
                    """ Historical observation is a concatenation of apple regrowth and cleaning action
                    trajectories, both downsampled by a factor of two. """
                    obs = torch.zeros(self.args.num_parallel_games, self.principal_obs_length)
                    for i in range(len(principal_obs_list)):
                        regrowth_traj = principal_obs_list[i]["regrowth_trajectory"].T
                        num_cleaned_traj = principal_obs_list[i]["num_cleaned_trajectory"].T
                        downsampled_regrowth = regrowth_traj[:, :: 2]
                        downsampled_num_cleaned = num_cleaned_traj[:, :: 2]
                        obs += torch.cat([downsampled_regrowth, downsampled_num_cleaned], dim=-1)
                    obs /= len(principal_obs_list)
        else:
            match self.args.env_name:
                case "cer":
                    obs = torch.full((self.args.num_parallel_games, self.principal_obs_length-3), 1).float()
                case "commons_harvest__open":
                    obs = torch.full((self.args.num_parallel_games, self.principal_obs_length), 1).float()
                case "clean_up":
                    obs = torch.full((self.args.num_parallel_games, self.principal_obs_length), 1).float()

        return obs

    def set_tax_vals(self, indicator, principal_step):
        """ Decide and set tax values / incentives. 

        Args:
            indicator (Tensor): one-hot encoding of environment indicator (e.g. lever activated
                                for CER, and empty if not applicable so concatenation adds nothing)
            principal_step (int): current principal step number

        Returns:
            (Tensor): tax rates / incentives chosen
        """     
        
        """ Extract historical observation to act on from information principal is allowed to observe. """
        historical_obs = self._process_observation(self.cached_principal_obs_list)
        
        """ Save observation as a field as a one-step buffer for optimisation later. """
        observation_to_act_on = torch.cat([indicator, historical_obs], dim=1).to(self.device)

        """ Produce output from policy net. This is a one-step trajectory we will update from in optimization later. """
        with torch.no_grad():
            # this will already be on device and we leave it there for optimisation steps after episode.
            self.tax_decision_data: TensorDict = self.principal_agent(observation_to_act_on)
        
        """ Store observation in one-step buffer for optimization. """
        self.tax_decision_data["obs"] = observation_to_act_on

        """ Retrieve action sampled from policy net. """
        action = self.tax_decision_data["actions"].cpu()

        """ Mask out final action, which is a NO-OP corresponding to leaving a tax rate unchanged. """
        no_op_mask = action == len(self.args.discretization_set)

        # add a dummy element to end of discretization set for NO-OP indices
        unmasked_tax_vals_per_game = torch.tensor(self.args.discretization_set + [0])[action]

        """ Apply no-op mask to repeat previous tax values where needed. """
        proposed_tax_vals_per_game = torch.where(
            no_op_mask, self.previous_proposed_tax_rates, unmasked_tax_vals_per_game
        )

        """ Store proposed tax rates for use with NO-OPs next time. """
        self.previous_proposed_tax_rates = proposed_tax_vals_per_game

        """ Return the tax rates that we set in environments. """
        return proposed_tax_vals_per_game

    def update(self, ctx, principal_step, episode_buffers):
        """ Update principal parameters using validation episodes.

        Args:
            ctx (Context): training context
            principal_step (int): principal step number
            episode_buffers (list[ValidationEpisode]): list of validation episode buffers
        """

        self.cached_principal_obs_list = [ep.get_episode_principal_observation() for ep in episode_buffers]

        # shape (num_val_episodes, num_parallel_games) - for measurements across validation episodes, use mean dim=0
        principal_rewards_per_game = torch.stack([ep.principal_cumulative_reward for ep in episode_buffers]).mean(dim=0)

        step_principal_agent_nets(
            principal_agent=self.principal_agent,
            optimizer=self.optimizer,
            principal_rewards_per_game=principal_rewards_per_game,
            tax_decision_data=self.tax_decision_data,
            ctx=ctx,
            args=self.args,
        )

    def save_params(self, principal_step):
        torch.save(
            self.principal_agent.state_dict(),
            f"./saved_params/step{principal_step}/seed_{self.args.seed}_principal_agent_net.pt",
        )


class LLMPrincipal(Principal):
    def __init__(self, args, num_brackets, envs):
        super().__init__(args)
        assert args.num_parallel_games == 1, "LLM method configured for one parallel game only."
        self.num_brackets = num_brackets
        self.demonstrations = ""
        self.prompt_backbone = get_prompt_backbone(args.llm_prompt_style)
        self.api_handler = api_handler_factory(args.llm_model, args.temperature)

    def _process_observation(self, principal_observation):
        obs = {}
        match self.args.env_name:
            case "commons_harvest__open":
                obs['mean_cumulative_raw_rewards'] = torch.stack(
                    list(principal_observation[i]["cumulative_agent_raw_rewards"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['mean_apple_trajectory'] = torch.stack(
                    list(principal_observation[i]["apples_trajectory"] for i in range(len(principal_observation)))
                ).mean(dim=0)
            case "clean_up":
                obs['mean_cumulative_raw_rewards'] = torch.stack(
                    list(principal_observation[i]["cumulative_agent_raw_rewards"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['mean_cumulative_num_cleaned'] = torch.stack(
                    list(principal_observation[i]["num_cleaned"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['mean_regrowth_trajectory'] = torch.stack(
                    list(principal_observation[i]["regrowth_trajectory"] for i in range(len(principal_observation)))
                ).mean(dim=0)
            case "cer":
                obs['num_lever0_per_step'] = torch.stack(
                    list(principal_observation[i]["num_lever0_per_step"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['num_lever1_per_step'] = torch.stack(
                    list(principal_observation[i]["num_lever1_per_step"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['num_lever2_per_step'] = torch.stack(
                    list(principal_observation[i]["num_lever2_per_step"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['num_door_per_step'] = torch.stack(
                    list(principal_observation[i]["num_door_per_step"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['num_start_per_step'] = torch.stack(
                    list(principal_observation[i]["num_start_per_step"] for i in range(len(principal_observation)))
                ).mean(dim=0)
                obs['door_open'] = torch.stack(
                    list(principal_observation[i]["door_open"] for i in range(len(principal_observation)))
                ).mean(dim=0)
        return obs
    
    def set_tax_vals(self, indicator, principal_step):
        """ Build the prompt and return the generated tax rate.

        Args:
            indicator_tensor (Tensor): one-hot encoding of environment indicator
            principal_step (int): current principal step number

        Returns:
            (Tensor): tax rates / incentives chosen
        """
        if indicator.shape[-1] == 0:
            indicator_prompt = ""
        else:
            indicator_int = indicator.argmax(dim=-1).item()
            indicator_prompt = f"Indicator {indicator_int} is activated. "

        prompt = (
            self.prompt_backbone["general_explanation"]
            + self.prompt_backbone["provide_history"]
            + self.demonstrations
            + self.prompt_backbone["reminder"]
            + indicator_prompt
        )
        if principal_step == 0:
            prompt += "This is your first attempt. "

        """ Decide a new tax rate and store it in current tax rate field. """
        # has shape (num_parallel_games, num_brackets), which here is (1, num_brackets)
        # *decorator requires prompt passed as a kwarg*
        self.current_tax_tensor = self.generate_and_log_response(principal_step, prompt=prompt)

        return self.current_tax_tensor

    @retry_with_prompt_adjustment(max_attempts=3)
    def generate_and_log_response(self, principal_step, prompt):
        """ Generate tax rates using the LLM and log the prompt and response.

            Args:
                prompt (str): prompt to send to the LLM - *decorator neeeds this passed as a kwarg*
                principal_step (int): current principal step number

            Returns:
                (Tensor): tax rates / incentives chosen

            Raises:
                Exception: If the response cannot be parsed and validated after retries.
            """
        unvalidated_repsonse = self.api_handler.query_llm(prompt)
        tax_rate = self.api_handler.parse_and_validate_response(unvalidated_repsonse, self.num_brackets)
        self.logger.log_prompt_and_response(prompt, unvalidated_repsonse, principal_step)
        return torch.tensor(tax_rate, dtype=torch.float32)

    def update(self, ctx, principal_step, episode_buffers):
        """ Update demonstrations using validation episodes.

        Args:
            ctx (Context): training context
            principal_step (int): principal step number
            episode_buffers (list[ValidationEpisode]): list of validation episode buffers
        """

        # assuming only one parallel game so can take mean without specifying dimension
        mean_reward = torch.stack([ep.principal_cumulative_reward for ep in episode_buffers]).mean().item()
        principal_obs_list = [ep.get_episode_principal_observation() for ep in episode_buffers]

        historical_obs = self._process_observation(principal_obs_list)

        latest_demonstration = format_demonstration(
            env_name=self.args.env_name,
            gets_historical_obs=self.args.principal_gets_historical,
            generation_num=principal_step+1,
            action=self.current_tax_tensor[0].numpy(),
            reward=mean_reward,
            historical_obs=historical_obs,
        )
        self.demonstrations += latest_demonstration
