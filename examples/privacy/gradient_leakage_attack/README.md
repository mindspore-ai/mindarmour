# 梯度逆向攻击（Gradient Inversion Attack）

使用 MindSpore 框架实现梯度逆向攻击（威胁模型可见参考文献[1]）.

我们提供了3种梯度逆向攻击算法（InvGrad, SeeThrough, StepWise，算法细节见参考文献[2-4]）、3种数据类型以及2种模型组合的攻击实例。

同时，使用者可以参考我们提供的攻击测试实例，并基于我们提供的函数，快速扩展出更多的测试案例。

除此之外，我们还提供了差分隐私、梯度裁剪等防御方法防御梯度泄露攻击的功能。

## 配置

### 依赖

环境配置请参考 `minimal_environment.yml`。

### 硬件

我们建议使用GPU来运行这些代码。

## 使用

请在使用前准备好数据集。

### 数据集

你可以使用公用的机器学习数据集，如: *CIFAR-100*, *Tiny-ImageNet*. 请在使用前传入正确的数据路径 `--data_path`.

或者你也可以使用自制的图像数据（224x224 px），请将其放置于文件夹 */custom_data/1_img/* 中。

### 运行攻击

我们提供了用户友好的运行方式。
如果你想要运行攻击（with default configuration），只需要在终端运行：

```shell
python main.py
```

或者你可以传入更多的参数: `--out_put`, `--data_path`, `--dataset`, `--model`, `--alg_name`, `--defense`, `--num_data_points`, `--max_iterations`,
`--step_size`, `--TV_scale`, `--TV_start`, `--BN_scale`, `--BN_start` and `--callback`.

| argument        | description                                                              |
|-----------------|--------------------------------------------------------------------------|
| out_put         | str: 输出路径                                                                |
| data_path       | str: 数据集的路径                                                              |
| dataset         | str: 'TinyImageNet', 'CIFAR100', 'WebImage'                              |
| model           | str: 'resnet18', 'resnet34'                                              |
| alg_name        | str: 'InvGrad', 'SeeThrough', 'StepWise'                                 |
| defense         | str: 'None', 'Vicinal Augment', 'Differential Privacy', 'Gradient Prune' |
| num_data_points | int: 同时重构的数据量                                                            |
| max_iterations  | int: 攻击最大迭代次数                                                            |
| step_size       | float: 攻击时的优化步长                                                          |
| TV_scale        | float: Total Variation 正则项权重                                             |
| TV_start        | int: Total Variation 开始时的步数                                              |
| BN_scale        | float: Batch Normalization 正则项权重                                         |
| TV_start        | int: Batch Normalization 开始时的步数                                          |
| callback        | int: 每经过该步数输出一次攻击结果                                                      |

## 开源协议

请参见文件: `LICENSE.md`.

## 参考文献

[1] Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." in NeurIPS, 2019.

[2] Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?." in NeurIPS, 2020.

[3] Yin, Hongxu, et al. "See through gradients: Image batch recovery via gradinversion." in CVPR, 2021.

[4] Ye, Zipeng, et al. "High-Fidelity Gradient Inversion in Distributed Learning." in AAAI, 2024.

