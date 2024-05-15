"""
Shadow Attack Alternative Network
"""
import collections
from mindspore import nn


class LeNetAlternativeConv1(nn.Cell):
    """
    LeNet Conv1
    """
    def __init__(self):
        super(LeNetAlternativeConv1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv0)
        self.layerdict['conv0'] = self.conv0

        self.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeConv1Arch2(nn.Cell):
    """
    LeNet Conv1
    """
    def __init__(self):
        super(LeNetAlternativeConv1Arch2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv0)
        self.layerdict['conv0'] = self.conv0

        self.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeReLU1(nn.Cell):
    """
    LeNet ReLU1
    """
    def __init__(self):
        super(LeNetAlternativeReLU1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu1 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativePool1(nn.Cell):
    """
    LeNet Pool1
    """
    def __init__(self):
        super(LeNetAlternativePool1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu1 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeConv2(nn.Cell):
    """
    LeNet Conv2
    """
    def __init__(self):
        super(LeNetAlternativeConv2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu1 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )
        self.features.append(self.conv2)
        self.layerdict['conv2'] = self.conv2

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeReLU2(nn.Cell):
    """
    LeNet ReLU2
    """
    def __init__(self):
        super(LeNetAlternativeReLU2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu1 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )
        self.features.append(self.conv2)
        self.layerdict['conv2'] = self.conv2

        self.relu2 = nn.ReLU()
        self.features.append(self.relu2)
        self.layerdict['relu2'] = self.relu2

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativePool2(nn.Cell):
    """
    LeNet Pool2
    """
    def __init__(self):
        super(LeNetAlternativePool2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu1 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )
        self.features.append(self.conv2)
        self.layerdict['conv2'] = self.conv2

        self.relu2 = nn.ReLU()
        self.features.append(self.relu2)
        self.layerdict['relu2'] = self.relu22

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool2)
        self.layerdict['pool2'] = self.pool2

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeFc1(nn.Cell):
    """
    LeNet Fc1
    """
    def __init__(self):
        super(LeNetAlternativeFc1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu1 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )
        self.features.append(self.conv2)
        self.layerdict['conv2'] = self.conv2

        self.relu2 = nn.ReLU()
        self.features.append(self.relu2)
        self.layerdict['relu2'] = self.relu2

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool2)
        self.layerdict['pool2'] = self.pool2

        self.feature_dim = 16 * 5 * 5
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.classifier.append(self.fc1)
        self.layerdict['fc1'] = self.fc1

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(-1, self.feature_dim)
        for layer in self.classifier:
            x = layer(x)
        return x


class LeNetAlternativeFc1Act(nn.Cell):
    """
    LeNet Fc1Act
    """
    def __init__(self):
        super(LeNetAlternativeFc1Act, self).__init__()
        self.features = []
        self.classifier = []
        self.layerdict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu1 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )
        self.features.append(self.conv2)
        self.layerdict['conv2'] = self.conv2

        self.relu2 = nn.ReLU()
        self.features.append(self.relu2)
        self.layerdict['relu2'] = self.relu2

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool2)
        self.layerdict['pool2'] = self.pool2

        self.feature_dim = 16 * 5 * 5
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.classifier.append(self.fc1)
        self.layerdict['fc1'] = self.fc1

        self.fc1act = nn.ReLU()
        self.classifier.append(self.fc1act)
        self.layerdict['fc1act'] = self.fc1act

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(-1, self.feature_dim)
        for layer in self.classifier:
            x = layer(x)
        return x


class CIFAR10CNNAlternativeConv11(nn.Cell):
    """
    CIFAR10CNN Alternative Conv11 network
    """
    def __init__(self):
        super(CIFAR10CNNAlternativeConv11, self).__init__()
        self.features = []
        self.layerdict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv11)
        self.layerdict['conv11'] = self.conv11

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getlayeroutput(self, x, targetlayer):
        targetlayer = self.layerdict.get(targetlayer, None)
        for layer in self.features:
            x = layer(x)
            if layer == targetlayer:
                return x
        msg = "Target Layer Error !"
        raise ValueError(msg)


class CIFAR10CNNAlternativeConv11Arch2(nn.Cell):
    """
    CIFAR10CNN Alternative Conv11Arch2 network
    """
    def __init__(self):
        super(CIFAR10CNNAlternativeConv11Arch2, self).__init__()
        self.features = []
        self.layerdict = collections.OrderedDict()

        self.conv10 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv10)
        self.layerdict['conv10'] = self.conv10

        self.conv11 = nn.Conv2d(
            in_channels=16,
            out_channels=64,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv11)
        self.layerdict['conv11'] = self.conv11

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getlayeroutput(self, x, targetlayer):
        targetlayer = self.layerdict.get(targetlayer, None)
        for layer in self.features:
            x = layer(x)
            if layer == targetlayer:
                return x
        msg = "Target Layer Error !"
        raise ValueError(msg)


class CIFAR10CNNAlternativeReLU22(nn.Cell):
    """
    CIFAR10CNN Alternative ReLU22 network
    """
    def __init__(self):
        super(CIFAR10CNNAlternativeReLU22, self).__init__()
        self.features = []
        self.layerdict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv1)
        self.layerdict['conv1'] = self.conv1

        self.relu11 = nn.ReLU()
        self.features.append(self.relu1)
        self.layerdict['relu1'] = self.relu1

        self.conv12 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv12)
        self.layerdict['conv12'] = self.conv12

        self.relu12 = nn.ReLU()
        self.features.append(self.relu12)
        self.layerdict['relu12'] = self.relu12

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv21 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv21)
        self.layerdict['conv21'] = self.conv21

        self.relu21 = nn.ReLU()
        self.features.append(self.relu21)
        self.layerdict['relu21'] = self.relu21

        self.conv22 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv22)
        self.layerdict['conv22'] = self.conv22

        self.relu22 = nn.ReLU()
        self.features.append(self.relu22)
        self.layerdict['relu22'] = self.relu22

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getlayeroutput(self, x, targetlayer):
        targetlayer = self.layerdict.get(targetlayer, None)
        for layer in self.features:
            x = layer(x)
            if layer == targetlayer:
                return x
        msg = "Target Layer Error !"
        raise ValueError(msg)


class CIFAR10CNNAlternativeReLU22Arch2(nn.Cell):
    """
    CIFAR10CNN Alternative ReLU22Arch2 network structure
    """
    def __init__(self):
        super(CIFAR10CNNAlternativeReLU22Arch2, self).__init__()
        self.features = []
        self.layerdict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=5,
            pad_mode='pad',
            padding=2,
        )
        self.features.append(self.conv11)
        self.layerdict['conv11'] = self.conv11

        self.relu11 = nn.ReLU()
        self.features.append(self.relu11)
        self.layerdict['relu11'] = self.relu11

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv22 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            pad_mode='pad',
            padding=2,
        )
        self.features.append(self.conv22)
        self.layerdict['conv22'] = self.conv22

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getlayeroutput(self, x, targetlayer):
        targetlayer = self.layerdict.get(targetlayer, None)
        for layer in self.features:
            x = layer(x)
            if layer == targetlayer:
                return x
        msg = "Target Layer Error !"
        raise ValueError(msg)


class CIFAR10CNNAlternativeReLU32(nn.Cell):
    """
    CIFAR10CNN Alternative ReLU32 network structure
    """
    def __init__(self):
        super(CIFAR10CNNAlternativeReLU32, self).__init__()
        self.features = []
        self.layerdict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv11)
        self.layerdict['conv11'] = self.conv11

        self.relu11 = nn.ReLU()
        self.features.append(self.relu11)
        self.layerdict['relu11'] = self.relu11

        self.conv12 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv12)
        self.layerdict['conv12'] = self.conv12

        self.relu12 = nn.ReLU()
        self.features.append(self.relu12)
        self.layerdict['relu12'] = self.relu12

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features.append(self.pool1)
        self.layerdict['pool1'] = self.pool1

        self.conv21 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv21)
        self.layerdict['conv21'] = self.conv21

        self.relu21 = nn.ReLU()
        self.features.append(self.relu21)
        self.layerdict['relu21'] = self.relu21

        self.conv22 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv22)
        self.layerdict['conv22'] = self.conv22

        self.relu22 = nn.ReLU()
        self.features.append(self.relu22)
        self.layerdict['relu22'] = self.relu22

        self.conv31 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv31)
        self.layerdict['conv31'] = self.conv31

        self.relu31 = nn.ReLU()
        self.features.append(self.relu31)
        self.layerdict['relu31'] = self.relu31

        self.conv32 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
        )
        self.features.append(self.conv32)
        self.layerdict['conv32'] = self.conv32

        self.relu32 = nn.ReLU()
        self.features.append(self.relu32)
        self.layerdict['relu32'] = self.relu32

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getlayeroutput(self, x, targetlayer):
        targetlayer = self.layerdict.get(targetlayer, None)
        for layer in self.features:
            x = layer(x)
            if layer == targetlayer:
                return x
        msg = "Target Layer Error !"
        raise ValueError(msg)
