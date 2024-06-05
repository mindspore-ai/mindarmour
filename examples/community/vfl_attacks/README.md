# 纵向联邦学习攻击

本项目基于MindSpore框架实现了纵向联邦学习模型的一种后门攻击方法和一种标签推理攻击方法。

## 原型论文

T. Zou, Y. Liu, Y. Kang, W. Liu, Y. He, Z. Yi, Q. Yang, and Y.-Q. Zhang, “Defending batch-level label inference and replacement attacks in vertical federated learning,” IEEE Transactions on Big Data, pp. 1–12, 2022. [PDF][https://www.computer.org/csdl/journal/bd/5555/01/09833321/1F8uKhxrvNe]

Fu C, Zhang X, Ji S, et al. Label inference attacks against vertical federated learning[C]//31st USENIX security symposium (USENIX Security 22). 2022: 1397-1414. [PDF][https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong]

## 环境要求

Mindspore >= 1.9

## 脚本说明

```markdown
│  README.md
│  the_example.py  // 应用示例
│
├─examples  //示例
│  ├─common
│  │  │  constants.py  //用户定义常量
│  │
│  ├─datasets
│  │  │  cifar_dataset.py   //用户加载数据集
│  │  │  functions.py
│  │
│  └─model
│      │  init_active_model.py   //用户加载顶层模型
│      │  init_passive_model.py  //用户加载底层模型
│      │  resnet.py  //用户定义模型结构
│      │  resnet_cifar.py
│      │  top_model_fcn.py
│      │  vgg.py
│      │  vgg_cifar.py
│
├─utils  //实现VFL功能和两个算法
│  ├─config
│  │  │  args_process.py  //审查并处理用户传入的参数
│  │  │  config.yaml  //默认参数配置文件
│  │
│  ├─datasets  //定义VFL数据集加载方式
│  │  │  common.py
│  │  │  image_dataset.py
│  │
│  ├─methods
│  │  ├─direct_attack
│  │  │  │  direct_attack_passive_party.py  //定义直接标签推理攻击的攻击者对象
│  │  │  │  direct_attack_vfl.py  //定义直接标签推理攻击的VFL对象
│  │  │
│  │  └─g_r
│  │      │  g_r_passive_party.py  //定义梯度替换后门攻击的攻击者对象
│  │
│  ├─model
│  │  │  base_model.py  //VFL中模型的基本类
│  │
│  ├─party
│  │  │  active_party.py  //主动方对象
│  │  │  passive_party.py  //被动方对象
│  │
│  └─vfl
│      │  init_vfl.py  //初始化各参与方
│      │  vfl.py  //定义VFL对象，包括各类过程函数
```

## 引入相关包

```Python
from utils.vfl.init_vfl import Init_Vfl
from utils.vfl.vfl import VFL
from utils.methods.direct_attack.direct_attack_vfl import DirectVFL
from utils.config.args_process import argsments_function
```

## Init_Vfl介绍

该模块负责垂直联邦学习（VFL）中参与者的初始化，包括参与者的模型、参数和类。对于主动参与方，定义对象为VFLActiveModel，对于正常被动参与方，定义对象为VFLPassiveModel，对于梯度替换后门攻击，定义攻击者对象为GRPassiveModel，对于直接标签推理攻击，定义对象为DirectAttackPassiveModel。

# VFL介绍

该模块定义了VFL中各种过程函数，包括训练、预测、更新等。

# DirectVFL 介绍

该模块实现了直接标签推理攻击，在VFL类的基础上定义了直接标签推理攻击中的过程函数。

# argsments_function 介绍

该函数接受并审查用户的参数，并封装为utils中各个类支持的格式。其中，梯度替换攻击用于分割VFL场景，直接标签推理攻击只适用于不分割VFL场景。