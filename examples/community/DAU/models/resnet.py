"""
ResNet in MindSpore.
Converted from PyTorch version.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal, Constant, initializer

class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                         stride=stride, has_bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                         stride=stride, has_bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class _ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10, in_channel=3, zero_init_residual=False):
        super(_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1,
                              padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Dense(512 * block.expansion, num_classes)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Initialize weights
        self._initialize_weights()
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            self._zero_init_residual()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(*layers)
    
    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                                               cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
                cell.gamma.set_data(initializer('ones', cell.gamma.shape))
                cell.beta.set_data(initializer('zeros', cell.beta.shape))
    
    def _zero_init_residual(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, Bottleneck):
                cell.bn3.gamma.set_data(initializer('zeros', cell.bn3.gamma.shape))
            elif isinstance(cell, BasicBlock):
                cell.bn2.gamma.set_data(initializer('zeros', cell.bn2.gamma.shape))

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def ResNet(num, num_classes=10):
    if num == 18:
        return _ResNet(BasicBlock, [2,2,2,2], num_classes, zero_init_residual=True)
    elif num == 34:
        return _ResNet(BasicBlock, [3,4,6,3], num_classes, zero_init_residual=True)
    elif num == 50:
        return _ResNet(Bottleneck, [3,4,6,3], num_classes, zero_init_residual=True)
    elif num == 101:
        return _ResNet(Bottleneck, [3,4,23,3], num_classes, zero_init_residual=True)
    elif num == 152:
        return _ResNet(Bottleneck, [3,8,36,3], num_classes, zero_init_residual=True)
    else:
        raise NotImplementedError
