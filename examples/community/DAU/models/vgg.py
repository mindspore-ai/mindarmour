import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import math
from mindspore.common.initializer import Normal, HeNormal, Constant, initializer

class VGG(nn.Cell):
    def __init__(self, features, num_classes=43):
        super(VGG, self).__init__()
        self.features = features
        self.flatten = nn.Flatten()
        self.linear = nn.Dense(512, num_classes)
        self._initialize_weights()

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(initializer(Normal(math.sqrt(2. / n)),
                                    cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer('ones', cell.gamma.shape))
                cell.beta.set_data(initializer('zeros', cell.beta.shape))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(Normal(0.01),
                                    cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape))


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, 
                              pad_mode='pad', has_bias=not batch_norm)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(num, batch_norm=False, num_classes=10, **kwargs):
    if batch_norm is True:
        if num == 11:
            return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes, **kwargs)
        elif num == 13:
            return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes, **kwargs)
        elif num == 16:
            return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes, **kwargs)
        elif num == 19:
            return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes, **kwargs)
        else:
            raise NotImplementedError
    else:
        if num == 11:
            return VGG(make_layers(cfg['A']), num_classes=num_classes, **kwargs)
        elif num == 13:
            return VGG(make_layers(cfg['B']), num_classes=num_classes, **kwargs)
        elif num == 16:
            return VGG(make_layers(cfg['D']), num_classes=num_classes, **kwargs)
        elif num == 19:
            return VGG(make_layers(cfg['E']), num_classes=num_classes, **kwargs)
        else:
            raise NotImplementedError

def vgg11(num_classes=10, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(cfg['A']), num_classes=num_classes, **kwargs)
    return model

def vgg11_bn(num_classes=10, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes, **kwargs)
    return model

def vgg13(num_classes=10, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), num_classes=num_classes, **kwargs)
    return model

def vgg13_bn(num_classes=10, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes, **kwargs)
    return model

def vgg16(num_classes=10, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), num_classes=num_classes, **kwargs)
    return model

def vgg16_bn(num_classes=10, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes, **kwargs)
    return model

def vgg19(num_classes=10, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E']), num_classes=num_classes, **kwargs)
    return model

def vgg19_bn(num_classes=10, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes, **kwargs)
    return model