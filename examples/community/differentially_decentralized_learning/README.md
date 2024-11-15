# 自适应噪声机制的差分隐私分布式学习
本项目基于MindSpore框架对分布式学习实现差分隐私保证, 通过利用实时梯度范数来动态估计各节点本地SGD的敏感性, 能够实现添加动态衰减的高斯噪声方差, 从而提高模型的训练精度。


## 环境要求
- mindspore >= 1.9: 本算法 mindspore 的集合通信库
- openmpi >= 5.0.1: 本算法需要执行多进程并行训, 开启多进程的命令 mpirun 依赖于 openmpi 库


## 脚本说明
```
├── README.md
├── dataset //存放数据集的路径
├── model_load.py //模型定义加载函数
├── data_load.py //数据集加载函数
├── adaptive_private_decentralized_learning.py  //自适应噪声差分隐私分布式学习的主函数,对应 ada_ddl 算法
└── private_decentralized_learning.py  //基于常数噪声的差分隐私分布式学习的主函数,对应 priv_ddl 算法
```


## 引入相关包

```python
from model_load import resnet18
from data_load import create_dataset
```


## 启动脚本
1. 开启8个进程/节点执行分布式训练, 运行ada_ddl算法:
```shell
mpirun --allow-run-as-root -n 8 python ./adaptive_private_decentralized_learning.py
```

2. 开启8个进程/节点执行分布式训练, 运行priv_ddl算法:
```shell
mpirun --allow-run-as-root -n 8 python ./private_decentralized_learning.py
```

