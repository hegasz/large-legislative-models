from skopt import Optimizer
import torch
from large_legislative_models.principal import Principal


class GaussianRegression(Principal):
    """ A GP regression "principal". """

    def __init__(
        self, args, n_initial_points=10, acq_func="EI", acq_optimizer="sampling", initial_point_generator="grid"
    ):
        super().__init__(args)
        if args.num_parallel_games > 1 or args.env_name == "cer":
            # does not work for multiple parallel games or contextual bandit settings
            raise NotImplementedError
            
        """ Scikit-optimize optimizer. """
        self.opt = Optimizer(
            dimensions=[(0.0, args.upper_bound), (0.0, args.upper_bound), (0.0, args.upper_bound)],
            base_estimator="GP",
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            initial_point_generator=initial_point_generator,
        )

    def set_tax_vals(self, unused, unused2):    
        """ Query optimizer for an action. """
        self.suggestion = self.opt.ask()
        tax_vals_per_game = torch.Tensor([self.suggestion])

        """ Return chosen rates. """
        return tax_vals_per_game

    def update(self, ctx, principal_step, episode_buffers):
        # shape (num_val_episodes, num_parallel_games) - for measurements across validation episodes, use mean dim=0
        mean_principal_reward_per_game = torch.stack(
            [ep.principal_cumulative_reward for ep in episode_buffers]
        ).mean(dim=0)
        self.opt.tell(self.suggestion, mean_principal_reward_per_game.item())
