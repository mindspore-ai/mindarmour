"""Some attack optimizers configurations."""
from mindspore import nn


class StepLR:
    """step learning rate."""
    def __init__(self, step_size, max_iter, gamma=0.33):
        self.max_it = max_iter
        self.mile1, self.mile2, self.mile3 = max_iter//2.667, max_iter//1.6, max_iter//1.142
        self.lr0 = step_size
        self.lr1 = self.lr0 * gamma
        self.lr2 = self.lr1 * gamma
        self.lr3 = self.lr2 * gamma

    def __call__(self, cur_step):
        if cur_step < self.mile1:
            return self.lr0
        if cur_step < self.mile2:
            return self.lr1
        if cur_step < self.mile3:
            return self.lr2
        return self.lr3


def optimizer_lookup(params, optim_name, step_size, scheduler=None, max_iterations=10_000):
    """optimizer configs."""
    if optim_name.lower() == "adam":
        optimizer = nn.Adam([params], learning_rate=step_size)
    elif optim_name.lower() == "sgd":
        optimizer = nn.SGD([params], learning_rate=step_size, momentum=0.0)
    else:
        raise ValueError(f"Invalid optimizer {optim_name} given.")

    if scheduler == "step-lr":
        scheduler = StepLR(step_size, max_iterations)
    elif scheduler == "cosine-decay":
        scheduler = nn.CosineDecayLR(0.0, step_size, max_iterations)
    elif scheduler == "linear":
        scheduler = nn.PolynomialDecayLR(step_size, 0.0, max_iterations, power=1.0)
    return optimizer, scheduler
