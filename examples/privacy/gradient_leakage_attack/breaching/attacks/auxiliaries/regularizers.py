"""all regularizers."""
import mindspore as ms
from mindspore import nn
from .deepinversion import DeepInversionFeatureHook


class TotalVariation(nn.Cell):
    """TV regularizer."""
    def __init__(self, scale=0.1, inner_exp=1, outer_exp=1, tv_start=0, double_opponents=False, eps=1e-8):

        super().__init__()
        self.scale = scale
        self.inner_exp = float(inner_exp)
        self.outer_exp = float(outer_exp)
        self.start = tv_start
        self.eps = eps
        self.double_opponents = double_opponents

        grad_weight = ms.Tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]]).unsqueeze(0).unsqueeze(1)
        grad_weight = ms.ops.concat((ms.ops.transpose(grad_weight, (0, 1, 3, 2)), grad_weight), 0)
        self.groups = 6 if self.double_opponents else 3
        self.weight = ms.ops.concat([grad_weight] * self.groups, 0)
        self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, pad_mode='pad', padding=1, dilation=1, group=self.groups)
        self.conv.trainable_params()[0].set_data(self.weight)

    def initialize(self, models, *args, **kwargs):
        pass

    def construct_(self, tensor, *args):
        if args[0] < self.start:
            return 100

        diffs = self.conv(tensor)

        squares = (diffs.abs() + self.eps).pow(self.inner_exp)
        squared_sums = (squares[:, 0::2] + squares[:, 1::2]).pow(self.outer_exp)
        return squared_sums.mean() * self.scale


class NormRegularization(nn.Cell):
    """Norm regularizer."""
    def __init__(self, scale=0.1, pnorm=2.0, norm_start=0):
        super().__init__()
        self.scale = scale
        self.pnorm = pnorm
        self.start = norm_start

    def initialize(self, models, *args, **kwargs):
        pass

    def construct_(self, tensor, *args):
        if args[0] < self.start:
            return 100
        return 1 / self.pnorm * tensor.pow(self.pnorm).mean() * self.scale


class DeepInversion(nn.Cell):
    """DeepInversion regularizer."""
    def __init__(self, scale=0.1, first_bn_multiplier=10, second_bn_multiplier=10,
                 deep_inv_start=0, deep_inv_stop=3000):
        super().__init__()
        self.scale = scale
        self.first_bn_multiplier = first_bn_multiplier
        self.second_bn_multiplier = second_bn_multiplier
        self.start = deep_inv_start
        self.stop = deep_inv_stop
        self.losses = []

    def initialize(self, models):
        self.losses = []
        model = models[0]
        for _, module in model.cells_and_names():
            if isinstance(module, nn.BatchNorm2d):
                self.losses.append(DeepInversionFeatureHook(module))

    def construct_(self, tensor, *args):
        """calculate DeepInversion loss."""
        if isinstance(tensor, int):
            print('DeepInversion Regularization')
        if args[0] < self.start:
            return 100
        if args[0] > self.stop:
            return 0
        rescale = [self.first_bn_multiplier, self.second_bn_multiplier] + [1.0 for _ in range(len(self.losses) - 2)]
        feature_reg = 0
        feature_reg += sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.losses)])
        return self.scale * feature_reg


regularizer_lookup = dict(
    total_variation=TotalVariation,
    norm=NormRegularization,
    deep_inversion=DeepInversion,
)
