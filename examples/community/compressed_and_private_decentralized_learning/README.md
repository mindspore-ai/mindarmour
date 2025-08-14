# 通信压缩的差分隐私分布式学习
本项目基于MindSpore框架对分布式学习实现差分隐私保证, 并利用量化、随机稀疏化、Top-K稀疏化等方式对通信进行压缩, 从而提高分布式隐私训练算法的通信高效性。


## 环境要求
- mindspore >= 1.9: 本算法 mindspore 的集合通信库
- openmpi >= 5.0.1: 本算法需要执行多进程并行训, 开启多进程的命令 mpirun 依赖于 openmpi 库


## 脚本说明
```
├── README.md
├── dataset //存放数据集的路径
├── model_load.py //模型定义加载函数
├── data_load.py //数据集加载函数
├── quantified_private_decentralized_learning.py  //基于通信量化的差分隐私分布式学习的主函数，对应quant_ddl算法
├── random_sparsified_private_decentralized_learning.py  //基于通信随机稀疏化的差分隐私分布式学习的主函数，对应randspar_ddl算法
└── top_k_sparsified_private_decentralized_learning.py  //基于通信Tok-K稀疏化的差分隐私分布式学习的主函数，对应topkspar_ddl算法
```


## 引入相关包

```python
from model_load import resnet18
from data_load import create_dataset
```


## 启动脚本
1. 开启8个进程/节点执行分布式训练, 运行quant_ddl算法:
```shell
mpirun --allow-run-as-root -n 8 python ./quantified_private_decentralized_learning.py
```

2. 开启8个进程/节点执行分布式训练, 运行randspar_ddl算法:
```shell
mpirun --allow-run-as-root -n 8 python ./random_sparsified_private_decentralized_learning.py
```

2. 开启8个进程/节点执行分布式训练, 运行topkspar_ddl算法:
```shell
mpirun --allow-run-as-root -n 8 python ./top_k_sparsified_private_decentralized_learning.py
```

