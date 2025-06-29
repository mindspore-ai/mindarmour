"""
ResNet model implementation in mindspore
"""
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops

class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, has_bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            ])
        else:
            self.shortcut = nn.SequentialCell()

        self.relu = ops.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            ])
        else:
            self.shortcut = nn.SequentialCell()

        self.relu = ops.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
        
    

class _ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.hook_dict = {}
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = ops.ReLU()

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(layers)

    def construct(self, x):
        # out = self.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.pool(out)
        # out = self.flatten(out)
        # out = self.fc(out)
        # return out
        out = self.relu(self.bn1(self.conv1(x)))
        self.hook_dict['conv1'] = out
        out = self.layer1(out)
        self.hook_dict['layer1'] = out
        out = self.layer2(out)
        self.hook_dict['layer2'] = out
        out = self.layer3(out)
        self.hook_dict['layer3'] = out
        out = self.layer4(out)
        self.hook_dict['layer4'] = out
        out = self.pool(out)
        self.hook_dict['pool'] = out
        out = self.flatten(out)
        self.hook_dict['flatten'] = out
        out = self.fc(out)
        self.hook_dict['fc'] = out
        return out
    
    def get_feature(self, layer_name: str):
        if layer_name not in self.hook_dict:
            raise ValueError(f"Layer {layer_name} not found in hook_dict. You should run the model with input first")
        return self.hook_dict[layer_name]
    
    def clear_hook(self):
        self.hook_dict.clear()
        
def ResNet(num, num_classes=10):
    if num == 18:
        return _ResNet(BasicBlock, [2,2,2,2], num_classes)
    elif num == 34:
        return _ResNet(BasicBlock, [3,4,6,3], num_classes)
    elif num == 50:
        return _ResNet(Bottleneck, [3,4,6,3], num_classes)
    elif num == 101:
        return _ResNet(Bottleneck, [3,4,23,3], num_classes)
    elif num == 152:
        return _ResNet(Bottleneck, [3,8,36,3], num_classes)
    else:
        raise NotImplementedError
    
def ResNet18(num_classes=10):
    return ResNet(18, num_classes)

def ResNet34(num_classes=10):
    return ResNet(34, num_classes)

def ResNet50(num_classes=10):
    return ResNet(50, num_classes)

def ResNet101(num_classes=10):
    return ResNet(101, num_classes)

def ResNet152(num_classes=10):
    return ResNet(152, num_classes)

if __name__ == "__main__":
    #  test the ResNet model
    #  use cuda
    import mindspore as ms
    import numpy as np
    ms.set_context(device_target="GPU")
    model = ResNet18()
    
    # random set an input in 32 x 32 x 3 (cifar10) and batch size 1
    input = Tensor(np.random.randn(1, 3, 32, 32), ms.float32)
    output = model(input)
    print(output.shape)
    print(output)
