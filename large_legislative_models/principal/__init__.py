from abc import ABC, abstractmethod


class Principal(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def set_tax_vals(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def save_params(self, principal_step):
        """Default to saving nothing for principals with no state."""
        pass
