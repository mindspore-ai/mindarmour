# Application demos of privacy stealing and privacy protection

## Introduction

Although machine learning could obtain a generic model based on training data, it has been proved that the trained
 model may disclose the information of training data (such as the membership inference attack).
Differential privacy training is an effective method proposed to overcome this problem, in which Gaussian noise is
 added while training. There are mainly three parts for differential privacy(DP) training: noise-generating
 mechanism, DP optimizer and DP monitor. We have implemented a novel noise-generating mechanisms: adaptive decay
 noise mechanism. DP monitor is used to compute the privacy budget while training.
Suppress Privacy training is a novel method to protect privacy distinct from the noise addition method
 (such as DP), in which the negligible model parameter is removed gradually to achieve a better balance between
 accuracy and privacy.

## 1. Adaptive decay DP training

With adaptive decay mechanism, the magnitude of the Gaussian noise would be decayed as the training step grows, which
 resulting a stable convergence.

```sh
cd examples/privacy/diff_privacy
python lenet5_dp_ada_gaussian.py
```

## 2. Adaptive norm clip training

With adaptive norm clip mechanism, the norm clip of the gradients would be changed according to the norm values of
 them, which can adjust the ratio of noise and original gradients.

```sh
cd examples/privacy/diff_privacy
python lenet5_dp.py
```

## 3. Membership inference evaluation

By this evaluation method, we could judge whether a sample is belongs to training dataset or not.

```sh
cd examples/privacy/membership_inference_attack
python train.py --data_path home_path_to_cifar100 --ckpt_path ./
python example_vgg_cifar.py --data_path home_path_to_cifar100 --pre_trained 0-100_781.ckpt
```

## 4. Suppress privacy training

With suppress privacy mechanism, the values of some trainable parameters  (such as conv layers and fully connected
 layers) are set to zero as the training step grows, which can
 achieve a better balance between accuracy and privacy

```sh
cd examples/privacy/sup_privacy
python sup_privacy.py
```

## 5. Image inversion attack

Inversion attack means reconstructing an image based on its deep representations. For example,
reconstruct a MNIST image based on its output through LeNet5. The mechanism behind it is that well-trained
model can "remember" those training dataset. Therefore, inversion attack can be used to estimate the privacy
leakage of training tasks.

```sh
cd examples/privacy/inversion_attack
python mnist_inversion_attack.py
```