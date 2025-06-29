# 复杂对抗环境下抗篡改抗窃取的分布式学习框架-训练阶段防篡改

本项目基于MindSpore框架实现Decoupling Adversarial Unlearning（DAU）算法，以实现在训练阶段防止模型被后门篡改的防御目标。

## 环境要求
- mindspore >= 2.5.0

## 脚本说明
```
├── README.md
├── config  //存放数据集的路径
├── core     //核心算法的实现目录
├── core/attacks //攻击算法的实现目录
├── models //模型定义
├── test //测试逻辑的入口
├── utils.py  // 工具代码实现

```


## 算法分为两步

### 1. anti_learning：过滤出可疑的有毒样本

**命令：**
```bash
python test_DAU.py --subtask "anti_learning" --dataset "dataset"

python test_DAU.py --subtask "anti_learning" --dataset "CIFAR-10"
```

### 2. unlearning：对可疑有毒样本进行unlearning，获得纯化模型

**命令：**
```bash
python test_DAU.py --subtask "unlearning" --dataset "dataset"

python test_DAU.py --subtask "unlearning" --dataset "CIFAR-10"
```
