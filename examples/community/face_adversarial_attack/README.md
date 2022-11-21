
# 人脸识别物理对抗攻击

## 描述

本项目是基于MindSpore框架对人脸识别模型的物理对抗攻击，通过生成对抗口罩，使人脸佩戴后实现有目标攻击和非目标攻击。

## 模型结构

采用华为MindSpore官方训练的FaceRecognition模型
https://www.mindspore.cn/resources/hub/details?MindSpore/1.7/facerecognition_ms1mv2

## 环境要求

mindspore>=1.7，硬件平台为GPU。

## 脚本说明

```markdown
├── readme.md
├── photos
│   ├── adv_input //对抗图像
│   ├── input //输入图像
│   └── target //目标图像
├── outputs //训练后的图像
├── adversarial_attack.py  //训练脚本
│── example_non_target_attack.py  //无目标攻击训练
│── example_target_attack.py  //有目标攻击训练
│── loss_design.py  //训练优化设置
└── test.py  //评估攻击效果
```

## 模型调用

方法一:

```python
#基于mindspore_hub库调用FaceRecognition模型
import mindspore_hub as mshub
from mindspore import context
def get_model():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
    model = "mindspore/1.7/facerecognition_ms1mv2"
    network = mshub.load(model)
    network.set_train(False)
    return network
```

方法二:

```text
利用MindSpore代码仓中的 <https://gitee.com/mindspore/models/blob/master/official/cv/FaceRecognition/eval.py> 的get_model函数加载模型
```

## 训练过程

有目标攻击:

```shell
cd face_adversarial_attack/
python example_target_attack.py
```

非目标攻击:

```shell
cd face_adversarial_attack/
python example_non_target_attack.py
```

## 默认训练参数

optimizer=adam, learning rate=0.01, weight_decay=0.0001, epoch=2000

## 评估过程

评估方法一:

```shell
adversarial_attack.FaceAdversarialAttack.test_non_target_attack()
adversarial_attack.FaceAdversarialAttack.test_target_attack()
```

评估方法二:

```shell
cd face_adversarial_attack/
python test.py
 ```

## 实验结果

有目标攻击:

```text
input_label: 60
target_label: 345
The confidence of the input image on the input label: 26.67
The confidence of the input image on the target label: 0.95
================================
adversarial_label: 345
The confidence of the adversarial sample on the correct label: 1.82
The confidence of the adversarial sample on the target label: 10.96
input_label: 60, target_label: 345, adversarial_label: 345

photos中是有目标攻击的实验结果
```

非目标攻击:

```text
input_label: 60
The confidence of the input image on the input label: 25.16
================================
adversarial_label: 251
The confidence of the adversarial sample on the correct label: 9.52
The confidence of the adversarial sample on the adversarial label: 60.96
input_label: 60, adversarial_label: 251
```
