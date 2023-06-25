# 使用Sensitivity Convergence Coverage测试深度学习模型

## 概述

传统软件的决策逻辑由代码逻辑决定，传统软件通过代码行覆盖率来判断当前测试是否充分，理想情况下覆盖率越高，代码测试越充分。然而，对于深度神经网络而言，程序的决策逻辑由训练数据、网络模型结构和参数通过某种黑盒机制决定，代码行覆盖率已不足以评估测试的充分性。需要根据深度网络的特点选择更为适合的测试评价准则，指导神经网络进行更为充分的测试，发现更多的边缘错误用例，从而确保模型的通用性、鲁棒性。

Sensitivity Convergence Coverage (SCC) 是一种基于神经元输出差异的深度学习模型测试方法。与神经元覆盖率（Neuron Coverage）关注神经元输出激活情况不同，SCC关注的是神经元输出之间的差异。具体来说，SCC关注的是神经元i在输入x和输入x'（x的噪声版本）下的输出差异，即$Neuron_{i}(x) - Neuron_{i}(x')$。

根据偏微分理论，任意神经元n的扰动分布$Neuron_{i}(x) - Neuron_{i}(x')$，其中x是输入，x'是噪声扰动输入，呈正态分布。这意味着大部分的噪声对神经元造成的扰动较小，只有少部分噪声对神经元造成较大的扰动。因此，我们可以通过抽样方法使扰动分布收敛，从而以较高概率发现所有的扰动和深度学习模型错误。

这里以LeNet模型，MNIST数据集为例，说明如何使用SCC测试深度学习模型。

## 实现阶段

### 需要导入的库文件

下列是我们需要的公共模块、MindSpore相关模块和fuzz_testing特性模块，以及配置日志标签和日志等级。

```python
import math
import numpy as np

from mindspore import Tensor
from mindarmour.fuzz_testing import CoverageMetrics
from mindspore.train.summary.summary_record import _get_summary_tensor_data

from mindarmour.utils._check_param import check_model, check_numpy_param, check_int_positive, \
    check_param_type, check_value_positive
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'SensitivityConvergenceCoverage'
```

### 参数配置

配置必要的信息，包括环境信息、执行的模式。

```python
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
```

详细的接口配置信息，请参见`set_context`接口说明。

### 计算Sensitivity Convergence Coverage 并使用SensitivityMaximizingFuzzer增大模型的SCC。

1. 建立LeNet模型，加载MNIST数据集，操作同[模型安全]()

```python
# Lenet model
model = Model(net)
# get training data
mnist_path = "../common/dataset/MNIST/"
batch_size = 32
ds = generate_mnist_dataset(os.path.join(mnist_path, "train"), batch_size, sparse=False)
train_images = []
for data in ds.create_tuple_iterator():
    images = data[0].asnumpy().astype(np.float32)
    train_images.append(images)
train_images = np.concatenate(train_images, axis=0)

# get test data
batch_size = 32
ds = generate_mnist_dataset(os.path.join(mnist_path, "test"), batch_size, sparse=False)
test_images = []
test_labels = []
for data in ds.create_tuple_iterator():
    images = data[0].asnumpy().astype(np.float32)
    labels = data[1].asnumpy()
    test_images.append(images)
    test_labels.append(labels)
test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

```

2. Coverage参数设置。

Sensitivity Convergence Coverage包含Model, threshold, batch_size, selected_neurons_num和n_iter。

Model：用户指定需要测试的模型。

threshold：神经元覆盖阈值，当覆盖率大于threshold时神经元覆盖率测试完成。

batch_size：测试过程中同时利用batch_size数量大小的输入计算Coverage。

selected_neurons_num：测试神经元数量，数量越大，测试越准确，但时间开销也越大。

n_iter：最大测试次数，避免模型长时间测试。

以下是Coverage参数配置例子：

```python
SCC = SensitivityConvergenceCoverage(model, t = 0.5, batch_size = 32)
```

3. Fuzzer参数设置。

设置数据变异方法及参数。支持同时配置多种方法，目前支持的数据变异方法包含两类：

自然扰动样本生成方法：

仿射变换类方法：Translate、Scale、Shear、Rotate、Perspective、Curve；

模糊类方法：GaussianBlur、MotionBlur、GradientBlur；

亮度调整类方法：Contrast、GradientLuminance;

加噪类方法：UniformNoise、GaussianNoise、SaltAndPepperNoise、NaturalNoise。

基于对抗攻击的白盒、黑盒对抗样本生成方法：FGSM（FastGradientSignMethod）、PGD（ProjectedGradientDescent）、MDIIM（MomentumDiverseInputIterativeMethod）。

数据变异方法中一定要包含基于图像像素值变化的方法。

前两种类型的图像变化方法，支持用户自定义配置参数，也支持算法随机选择参数。用户自定义参数配置范围请参考:https://gitee.com/mindspore/mindarmour/tree/master/mindarmour/natural_robustness/transform/image 中对应的类方法。算法随机选择参数，则params设置为'auto_param': [True]，参数将在推荐范围内随机生成。

基于对抗攻击方法的参数配置请参考对应的攻击方法类。

下面是变异方法及其参数配置的一个例子：

```python
mutate_config = [{'method': 'GaussianBlur',
                  'params': {'ksize': [1, 2, 3, 5], 'auto_param': [True, False]}},
                 {'method': 'MotionBlur',
                  'params': {'degree': [1, 2, 5], 'angle': [45, 10, 100, 140, 210, 270, 300],
                  'auto_param': [True]}},
                 {'method': 'UniformNoise',
                  'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
                 {'method': 'GaussianNoise',
                  'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
                 {'method': 'Contrast',
                  'params': {'alpha': [0.5, 1, 1.5], 'beta': [-10, 0, 10], 'auto_param': [False, True]}},
                 {'method': 'Rotate',
                  'params': {'angle': [20, 90], 'auto_param': [False, True]}},
                 {'method': 'FGSM',
                  'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1], 'bounds': [(0, 1)]}}]
```

初始化种子队列，种子队列中的每个种子，包含2个值：原始图片、图片标签。这里取100个样本作为初始种子队列。

```python
# make initial seeds
initial_seeds = []
for img, label in zip(test_images, test_labels):
    initial_seeds.append([img, label])
initial_seeds = initial_seeds[:100]
```

3. 实例化Sensitivity Convergence Coverage类，并计算初始覆盖率。

```python
SCC = SensitivityConvergenceCoverage(model, t = 0.5, batch_size = 32)
scc_value = SensitivityConvergenceCoverage.metrics(initial_seeds)
print('SCC of initial seeds is: ', scc_value)
```

结果：

```python
SCC of initial seeds is: 0.2969543147208122
```

4. 实例化SensitivityMaximizingFuzzer类, Fuzz生成test case提升覆盖率指标。

```python
model_fuzz_test = SensitivityMaximizingFuzzer(model)
fuzz_samples, gt_labels, preds, strategies, metrics = model_fuzz_test.fuzzing(mutate_config, initial_seeds, SCC, max_iters=10)
```

5. 实验结果。

fuzzing的返回结果中包含了5个数据：fuzz生成的样本fuzz_samples、生成样本的真实标签true_labels、被测模型对于生成样本的预测值fuzz_preds、 生成样本使用的变异方法fuzz_strategies、fuzz testing生成的test cases的metrics。用户可使用这些返回结果进一步的分析模型的鲁棒性。这里只展开metrics，查看fuzz testing后的各个评估指标。

```python
if metrics:
    for key in metrics:
        LOGGER.info(TAG, key + ': %s', metrics[key])
```

Fuzz测试后结果如下:

```python
Accuracy:  0.80
Attack_success_rate:  0.20
coverage_metrics:  0.35971389017074296
```