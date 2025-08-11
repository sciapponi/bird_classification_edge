import torch

class CombinedOptimizer:
    def __init__(self, main_optimizer: torch.optim.Optimizer, filter_optimizer: torch.optim.Optimizer):
        self.optimizers = {
            'main': main_optimizer,
            'filter': filter_optimizer
        }
        # Initially, both optimizers are active by default
        self.active_optimizers = ['main', 'filter']

    def set_active_optimizers(self, active_keys: list[str]):
        """
        Sets which optimizers are currently active.
        Args:
            active_keys: A list of keys for the optimizers to activate (e.g., ['main', 'filter']).
        """
        self.active_optimizers = active_keys

    def zero_grad(self, set_to_none: bool = False):
        """Zeroes the gradients of all active optimizers."""
        for key in self.active_optimizers:
            if key in self.optimizers:
                self.optimizers[key].zero_grad(set_to_none=set_to_none)

    def step(self):
        """Performs a single optimization step on all active optimizers."""
        for key in self.active_optimizers:
            if key in self.optimizers:
                self.optimizers[key].step()

    def state_dict(self):
        """Returns the state of all optimizers."""
        return {key: opt.state_dict() for key, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        """Loads the state of all optimizers."""
        for key, state in state_dict.items():
            if key in self.optimizers:
                self.optimizers[key].load_state_dict(state)
    
    @property
    def param_groups(self):
        """
        Returns the parameter groups of the main optimizer for logging and scheduler compatibility.
        """
        return self.optimizers['main'].param_groups

    @property
    def defaults(self):
        """
        Returns the default hyperparameters of the main optimizer.
        """
        return self.optimizers['main'].defaults 