# SEM Track_Minspore
本代码利用mindspore框架实现了SEM Track攻击

## 依赖
* 硬件：Atlas 800/Atlas 800T A2
* MindSpore：2.2.13
* CANN: 7.0
* MindFormers版本：1.0

## 攻击数据集
  数据通过MindFormers llama7b模型，使用alpaca_52k进行模型推理，使用第0层LLamaDecoderlayer构建embedding model，收集其输入输出10000条制作SEM Track数据集，实现SEM attack

## 运行SEM攻击

```
python sem_track.py --dataset_npz my_dataset_10_samples.npz
```
运行后截图

![image](https://github.com/user-attachments/assets/faa1b0a0-0ae2-4ca7-8700-0d20f1789331)

生成攻击模型权重

![image](https://github.com/user-attachments/assets/5d9398ad-c36d-4fb9-9445-b36271671270)

