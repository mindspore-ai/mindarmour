"""
Inversion Attack Network
"""
from mindspore import nn


class LeNetDecoderConv1(nn.Cell):
    """
    Conv1 -> Input
    """
    def __init__(self):
        super(LeNetDecoderConv1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=1,
            kernel_size=5,
            pad_mode='pad',
            padding=4)

    def construct(self, x):
        return self.conv1(x)


class LeNetDecoderReLU2(nn.Cell):
    """
    Conv2 -> ReLU -> Conv1 -> Input
    """
    def __init__(self):
        super(LeNetDecoderReLU2, self).__init__()
        self.decoder = []

        self.deconv1 = nn.Conv2dTranspose(
            in_channels=16,
            out_channels=6,
            kernel_size=5,
        )

        self.relu1 = nn.ReLU()

        self.deconv2 = nn.Conv2dTranspose(
            in_channels=6,
            out_channels=1,
            kernel_size=5,
            stride=3,
            pad_mode='pad',
            output_padding=0
        )

    def construct(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        return x


class CIFAR10CNNDecoderConv11(nn.Cell):
    """
    Conv1 -> Input
    """
    def __init__(self):
        super(CIFAR10CNNDecoderConv11, self).__init__()
        self.deconv1 = nn.Conv2dTranspose(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            pad_mode='pad',
            padding=1
        )

    def construct(self, x):
        x = self.deconv1(x)
        return x


class CIFAR10CNNDecoderReLU22(nn.Cell):
    """
    Conv22 -> ReLU22 -> Conv11 -> INPUT
    """
    def __init__(self):
        super(CIFAR10CNNDecoderReLU22, self).__init__()
        self.deconv11 = nn.Conv2dTranspose(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            pad_mode='pad',
            padding=1,
            output_padding=1
        )
        self.relu1 = nn.ReLU()
        self.deconv21 = nn.Conv2dTranspose(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            pad_mode='pad',
            padding=1
        )

    def construct(self, x):
        x = self.deconv11(x)
        x = self.relu1(x)
        x = self.deconv21(x)
        return x


class CIFAR10CNNDecoderReLU32(nn.Cell):
    """
    Conv33 -> ReLU22 -> Conv22 -> ReLU11 -> Conv11 -> Input
    """
    def __init__(self):
        super(CIFAR10CNNDecoderReLU32, self).__init__()
        self.deconv11 = nn.Conv2dTranspose(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            pad_mode='pad',
            output_padding=1
        )
        self.relu1 = nn.ReLU()
        self.deconv21 = nn.Conv2dTranspose(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            pad_mode='pad',
            output_padding=1
        )
        self.relu2 = nn.ReLU()

        self.deconv31 = nn.Conv2dTranspose(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            padding=1
        )

    def construct(self, x):
        x = self.deconv11(x)
        x = self.relu1(x)
        x = self.deconv21(x)
        x = self.relu2(x)
        x = self.deconv31(x)
        return x
