# MindArmour

<!-- TOC -->

- [MindArmour](#mindarmour)
    - [简介](#简介)
        - [对抗样本鲁棒性模块](#对抗样本鲁棒性模块)
        - [Fuzz Testing模块](#fuzz-testing模块)
        - [隐私保护模块](#隐私保护模块)
            - [差分隐私训练模块](#差分隐私训练模块)
            - [隐私泄露评估模块](#隐私泄露评估模块)
    - [开始](#开始)
        - [确认系统环境信息](#确认系统环境信息)
        - [安装](#安装)
            - [源码安装](#源码安装)
            - [pip安装](#pip安装)
        - [验证是否成功安装](#验证是否成功安装)
    - [文档](#文档)
    - [社区](#社区)
    - [贡献](#贡献)
    - [版本](#版本)
    - [版权](#版权)

<!-- /TOC -->

[View English](./README.md)

## 简介

MindArmour关注AI的安全和隐私问题。致力于增强模型的安全可信、保护用户的数据隐私。主要包含3个模块：对抗样本鲁棒性模块、Fuzz Testing模块、隐私保护与评估模块。

### 对抗样本鲁棒性模块

对抗样本鲁棒性模块用于评估模型对于对抗样本的鲁棒性，并提供模型增强方法用于增强模型抗对抗样本攻击的能力，提升模型鲁棒性。对抗样本鲁棒性模块包含了4个子模块：对抗样本的生成、对抗样本的检测、模型防御、攻防评估。

对抗样本鲁棒性模块的架构图如下：

![mindarmour_architecture](docs/adversarial_robustness_cn.png)

### Fuzz Testing模块

Fuzz Testing模块是针对AI模型的安全测试，根据神经网络的特点，引入神经元覆盖率，作为Fuzz测试的指导，引导Fuzzer朝着神经元覆盖率增加的方向生成样本，让输入能够激活更多的神经元，神经元值的分布范围更广，以充分测试神经网络，探索不同类型的模型输出结果和错误行为。

Fuzz Testing模块的架构图如下：

![fuzzer_architecture](docs/fuzzer_architecture_cn.png)

### 隐私保护模块

隐私保护模块包含差分隐私训练与隐私泄露评估。

#### 差分隐私训练模块

差分隐私训练包括动态或者非动态的差分隐私`SGD`、`Momentum`、`Adam`优化器，噪声机制支持高斯分布噪声、拉普拉斯分布噪声，差分隐私预算监测包含ZCDP、RDP。

差分隐私的架构图如下：

![dp_architecture](docs/differential_privacy_architecture_cn.png)

#### 隐私泄露评估模块

隐私泄露评估模块用于评估模型泄露用户隐私的风险。利用成员推理方法来推测样本是否属于用户训练数据集，从而评估深度学习模型的隐私数据安全。

隐私泄露评估模块框架图如下：

![privacy_leakage](docs/privacy_leakage_cn.png)

## 开始

### 确认系统环境信息

- 硬件平台为Ascend、GPU或CPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。  
    MindArmour与MindSpore的版本需保持一致。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/master/setup.py)。

### 安装

#### 源码安装

1. 从Gitee下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindarmour.git
    ```

2. 在源码根目录下，执行如下命令编译并安装MindArmour。

    ```bash
    cd mindarmour
    python setup.py install
    ```
 
3. 图片自然扰动算法使用到Perlin-numpy，安装：

    ```bash
    pip3 install git+https://github.com/pvigier/perlin-numpy
    ```

#### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindArmour/{arch}/mindarmour-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindArmour安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/master/setup.py)），其余情况需自行安装。
> - `{version}`表示MindArmour版本号，例如下载1.0.1版本MindArmour时，`{version}`应写为1.0.1。  
> - `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。

### 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindarmour'`，则说明安装成功。

```bash
python -c 'import mindarmour'
```

## 文档

安装指导、使用教程、API，请参考[用户文档](https://gitee.com/mindspore/docs)。

## 社区

社区问答：[MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/enQtOTcwMTIxMDI3NjM0LTNkMWM2MzI5NjIyZWU5ZWQ5M2EwMTQ5MWNiYzMxOGM4OWFhZjI4M2E5OGI2YTg3ODU1ODE2Njg1MThiNWI3YmQ)。

## 贡献

欢迎参与社区贡献，详情参考[Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 版本

版本信息参考：[RELEASE](RELEASE.md)。

## 版权

[Apache License 2.0](LICENSE)
