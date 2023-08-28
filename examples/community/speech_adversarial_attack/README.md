# 音频识别对抗噪声生成

## 描述

本项目是基于MindSpore框架实现音频识别模型对抗噪声生成算法[PAT](https://arxiv.org/pdf/2211.10661.pdf)，通过生成基于音素的通用对抗噪声来干扰音频识别模型的正常运行。

## 模型结构

采用启智平台开源的基于MindSpore框架的[DeepSpeech2模型](https://openi.pcl.ac.cn/Yzx835/speech_adversarial_attack)，模型参数文件可以从 https://openi.pcl.ac.cn/Yzx835/speech_adversarial_attack/modelmanage/show_model 里面获取。

## 环境要求

mindspore>=2.0.0a0，硬件平台为GPU、CPU、Ascend。

## 脚本说明

```markdown
├── attack.py //对抗噪声生成代码
├── checkpoints
│   └── DeepSpeech2_model.ckpt //DeepSpeech模型参数文件
├── data //数据路径
│   ├── libri_val_manifest.csv
│   └── LibriSpeech_dataset
│       └── val
│           ├── txt
│           └── wav
├── dataset.py //数据加载相关代码
├── labels.json //音频相关标签
├── README.md
├── requirements.txt
├── source
│   └── 100_test_audio_list.npy //固定的测试数据
├── src //DeepSpeech模型相关代码
│   ├── config.py
│   ├── deepspeech2.py
│   ├── greedydecoder.py
│   ├── __init__.py
└── stft.py //音频处理代码
```

## 训练依赖

环境依赖：

```python
numpy
easydict
librosa >= 0.8.1
soundfile >= 0.12.1
Levenshtein >= 0.20.9
g2p-en >= 2.1.0
mindspore >= 2.0.0a0
```

g2p-en 可能需要 nltk 工具中的 cmudict。 g2p-en运行的时候，nltk会自动下载依赖数据，如果下载失败，需要在官网（https://www.nltk.org/nltk_data/ ）手动下载cmudict.zip和averaged_perceptron_tagger.zip，然后在运行环境的home目录下创建 nltk_data/corpora/ 目录和 nltk_data/taggers/目录，然后把 cmudict.zip放在nltk_data/corpora/，把averaged_perceptron_tagger.zip放在nltk_data/taggers/。

模型：

将模型参数文件放入`checkpoints/`路径内，或修改`attack.py`代码中的相关路径：

```python
87: param_dict = load_checkpoint('./checkpoints/DeepSpeech2_model.ckpt')
```

数据：

将处理后的[librispeech](https://openi.pcl.ac.cn/Yzx835/speech_adversarial_attack/datasets)数据集存入`data/`路径内，具体参考`脚本说明`中的路径放置数据集，或修改`src/config.py`相关的路径参数：

```python
27:         "train_manifest": 'data/libri_train_manifest.csv',
80:         "test_manifest": 'data/libri_val_manifest.csv',
```

## 训练过程

噪声生成:

```shell
python attack.py
```

## 默认训练参数

```python
sample_rate = 16000
window_size = 0.02
window_stride = 0.01
window = "hamming"
adv_len=3200
eps=0.01
step=2467
iteration=30  # iteration=10 也可以达到攻击效果
lr=2e-3
```

## 实验结果

```text
攻击前： 错误率很低
SR = 0.0
CER = 0.036993539852976084

攻击后： 错误率提升
SR = 0.98
CER = 0.76
```