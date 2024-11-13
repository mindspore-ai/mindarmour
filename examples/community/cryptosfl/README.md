## 拆分联邦学习的函数加密防御

## 描述

本项目是基于MindSpore框架针对拆分联邦学习模型的防御手段，利用函数加密技术来保护拆分层输出，并保持模型的精度。

## 模型

客户端模型采用基于MindSpore框架的MLP模型，包含1层隐藏层，其输出维度与服务器的输入维度保持一致。

## 数据集

使用手写数字数据集MNIST作为训练和测试数据集。使用以下命令下载后保存在`datasets`文件夹中。

```bash
cd datasets/
python generate_data.py
```

## 加密配置

采用预生成的离散对数表加速函数加密的加密及解密过程。使用以下命令生成该表后保存在`config`文件夹中。

```bash
cd config/
python generate_config.py
```

## 原型论文

Xu R, Joshi J, Li C. Nn-emd: Efficiently training neural networks using encrypted multi-sourced datasets[J]. IEEE
Transactions on Dependable and Secure Computing, 2021, 19(4): 2807-2820.[[pdf]](https://arxiv.org/pdf/2012.10547)

Thapa C, Arachchige P C M, Camtepe S, et al. Splitfed: When federated learning meets split learning[C]//Proceedings of
the AAAI Conference on Artificial Intelligence. 2022, 36(8): 8485-8493.[[pdf]](https://arxiv.org/pdf/2004.12088)

## 环境要求

Mindspore >= 1.9，硬件平台为GPU/CPU/Ascend。

## 脚本说明

```markdown
├── README.md  
├── config  
│ ├── generate_config.py //生成函数加密所需的离散对数表  
├── crypto  
│ ├── sife_dynamic.py //基于内积的函数加密的实现  
│ ├── utils.py //函数加密实现所需的常用函数  
├── datasets  
│ ├── generate_data.py //加载数据的相关代码  
├── nn  
│ ├── smc.py //参与方加密及解密  
│ ├── worker.py //参与方模型训练及预测  
│ ├── utils.py //初始化参与方模型的函数  
└── example_cryptosfl.py //在拆分联邦学习上的应用例  
```

## 训练依赖

```markdown
numpy == 1.26.4  
gmpy2 == 2.1.5
```

## 训练过程

```bash
python example_cryptosfl.py
```

## 默认训练参数

```markdown
epochs = 20  
num_users = 5  
batch_size = 50  
model = 'mlp64'  
lr = 1e-3  
```

## 实验结果

```markdown
epoch 1, test accuracy = 93.50%  
epoch 2, test accuracy = 94.74%  
epoch 3, test accuracy = 95.43%  
epoch 4, test accuracy = 96.07%  
epoch 5, test accuracy = 96.22%  
epoch 6, test accuracy = 96.49%  
epoch 7, test accuracy = 96.77%  
epoch 8, test accuracy = 96.79%  
epoch 9, test accuracy = 97.02%  
epoch 10, test accuracy = 97.05%  
epoch 11, test accuracy = 97.19%  
epoch 12, test accuracy = 97.42%  
epoch 13, test accuracy = 97.59%  
epoch 14, test accuracy = 97.64%  
epoch 15, test accuracy = 97.65%  
epoch 16, test accuracy = 97.76%  
epoch 17, test accuracy = 97.86%  
epoch 18, test accuracy = 97.87%  
epoch 19, test accuracy = 97.95%  
epoch 20, test accuracy = 97.92%  
```

