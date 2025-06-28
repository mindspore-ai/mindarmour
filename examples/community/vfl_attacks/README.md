# 纵向联邦学习攻击

本项目基于MindSpore框架实现了纵向联邦学习模型的三种后门攻击方法和一种标签推理攻击方法。

## 原型论文

T. Zou, Y. Liu, Y. Kang, W. Liu, Y. He, Z. Yi, Q. Yang, and Y.-Q. Zhang, “Defending batch-level label inference and replacement attacks in vertical federated learning,” IEEE Transactions on Big Data, pp. 1–12, 2022. [PDF][https://www.computer.org/csdl/journal/bd/5555/01/09833321/1F8uKhxrvNe]

Fu C, Zhang X, Ji S, et al. Label inference attacks against vertical federated learning[C]//31st USENIX security symposium (USENIX Security 22). 2022: 1397-1414. [PDF][https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong]

Gu Y, Bai Y. LR-BA: Backdoor attack against vertical federated learning using local latent representations[J]. Computers & Security, 2023, 129: 103193. [PDF][https://www.sciencedirect.com/science/article/abs/pii/S0167404823001037]

Bai Y, Chen Y, Zhang H, et al. {VILLAIN}: Backdoor attacks against vertical split learning[C]//32nd USENIX Security Symposium (USENIX Security 23). 2023: 2743-2760. [PDF][https://www.usenix.org/conference/usenixsecurity23/presentation/bai]

## 环境要求

Mindspore >= 1.9

## 脚本说明

```markdown
│  README.md
│  example.py  // 应用示例
│
├─common
│  │  constants.py  //用户定义常量
│  │  image_report.py  //用户定义图像报告
│  │  parser.py  //用户定义参数解析器
│  │  utils.py  //用户定义工具函数
│
├─datasets
│  │  base_dataset.py   //VFL中模型的基本类
│  │  bhi_dataset.py   //用户加载数据集
│  │  cifar_dataset.py  //用户加载数据集
│  │  criteo_dataset.py  //用户加载数据集
│  │  cinic_dataset.py   //用户加载数据集
│  │  common.py
│  │  image_dataset.py
│  │  multi_image_dataset.py
│  │  tabular_dataset.py
│  │  multi_tabular_dataset.py
│
├─evaluate
│  ├─config
│  │  │  bhi.yaml  //默认参数配置文件
│  │  │  cifar10.yaml  //默认参数配置文件
│  │  │  cifar100.yaml  //默认参数配置文件
│  │  │  cinic.yaml  //默认参数配置文件
│  │  │  criteo.yaml  //默认参数配置文件
│  │
│  ├─args_line.py  //审查并处理用户传入的参数
│  ├─MainTask.py
│
├─methods
│  ├─direct_attack
│  │  │  direct_attack_passive_party.py  //定义直接标签推理攻击的攻击者对象
│  │  │  direct_attack_vfl.py  //定义直接标签推理攻击的VFL对象
│  │
│  ├─g_r
│  │  │  g_r_passive_party.py  //定义梯度替换后门攻击的攻击者对象
│  │
│  ├─villain
│  │  │  villain_passive_party.py  //定义直villain后门攻击的攻击者对象
│  │  │  villain_vfl.py  //定义villain后门攻击的VFL对象
│
├─model
│  │  base_model.py  //VFL中模型的基本类
│  │  init_active_model.py   //用户加载顶层模型
│  │  init_passive_model.py  //用户加载底层模型
│  │  resnet.py  //用户定义模型结构
│  │  resnet_cifar.py
│  │  top_model_fcn.py
│  │  bottom_model_fcn.py
│  │  vgg.py
│  │  vgg_cifar.py
│
├─party
│  │  active_party.py  //主动方对象
│  │  passive_party.py  //被动方对象
│
├─vfl
│  │  init_vfl.py  //初始化各参与方
│  │  vfl.py  //定义VFL对象，包括各类过程函数


```

## 引入相关包

```Python
from vfl.init_vfl import Init_Vfl
from vfl.vfl import VFL
from methods.direct_attack.direct_attack_vfl import DirectVFL
from methods.g_r.g_r_passive_party import GRPassiveModel
```

## Init_Vfl介绍

该模块负责垂直联邦学习（VFL）中参与者的初始化，包括参与者的模型、参数和类。对于主动参与方，定义对象为VFLActiveModel，对于正常被动参与方，定义对象为VFLPassiveModel，对于梯度替换后门攻击，定义攻击者对象为GRPassiveModel，对于直接标签推理攻击，定义对象为DirectAttackPassiveModel。

## VFL介绍

该模块定义了VFL中各种过程函数，包括训练、预测、更新等。

## DirectVFL 介绍

该模块实现了直接标签推理攻击，在VFL类的基础上定义了直接标签推理攻击中的过程函数。

## GRPassiveModel 介绍

该模块实现了梯度替换后门攻击，在VFLPassiveModel类的基础上定义了梯度替换后门攻击中的过程函数。

## VillainVFL 介绍

该模块实现了Villain后门攻击，在VFL类的基础上定义了Villain后门攻击中的过程函数。

## 扩展
本项目当前支持CIFAR-10、CIFAR100、BHI、criteo数据集，目前datasets文件夹中给出了相应数据集加载代码。如需自定义模型结构或数据集加载方式，请参考并修改datasets文件夹中的对应文件内容。
