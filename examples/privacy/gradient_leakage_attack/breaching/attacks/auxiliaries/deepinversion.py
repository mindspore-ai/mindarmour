"""Deepinversion Regurarizer."""
import mindspore as ms


class DeepInversionFeatureHook:
    """Deepinversion Regurarizer."""
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.r_feature = 0

    def hook_fn(self, ids, inputs, output):
        """hook function."""
        if isinstance(output, int):
            print(ids)
        nch = inputs[0].shape[1]
        mean = inputs[0].mean([0, 2, 3])
        var = ms.ops.transpose(inputs[0], (1, 0, 2, 3)).reshape([nch, -1]).var(1)

        r_feature = (ms.ops.norm(self.module.moving_variance.value() - var, 2) +
                     ms.ops.norm(self.module.moving_mean.value() - mean, 2))

        self.r_feature = r_feature

    def close(self):
        """remove hook."""
        self.hook.remove()
