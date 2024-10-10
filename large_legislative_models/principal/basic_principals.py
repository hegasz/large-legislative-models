import torch

from large_legislative_models.principal import Principal

class FixedTaxRate(Principal):
    """ "Principal" outputting one fixed set of tax rates / incentives. """

    def __init__(self, args, envs, fixed_rate):
        super().__init__(args)
        self.tax_vals_per_game = torch.Tensor(fixed_rate).repeat((self.args.num_parallel_games, 1))

    def set_tax_vals(self, indicator, ctx, envs):
        return self.tax_vals_per_game

    def update(self, *unused, **kwargs):
        # note I'd avoid using "*args" due to potential clash with the config, which is called args
        pass
