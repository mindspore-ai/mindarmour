# Application demos of privacy stealing and privacy protection
## Introduction
Although machine learning could obtain a generic model based on training data, it has been proved that the trained
 model may disclose the information of training data (such as the membership inference attack). Differential
  privacy training
  is an effective
  method proposed
  to overcome this problem, in which Gaussian noise is added while training. There are mainly three parts for
   differential privacy(DP) training: noise-generating mechanism, DP optimizer and DP monitor. We have implemented
    a novel noise-generating mechanisms: adaptive decay noise mechanism. DP
     monitor is used to compute the privacy budget while training.

## 1. Adaptive decay DP training
With adaptive decay mechanism, the magnitude of the Gaussian noise would be decayed as the training step grows, which
 resulting a stable convergence.
```sh
$ cd examples/privacy/diff_privacy
$ python lenet5_dp_ada_gaussian.py
```
## 2. Adaptive norm clip training
With adaptive norm clip mechanism, the norm clip of the gradients would be changed according to the norm values of
 them, which can adjust the ratio of noise and original gradients.
```sh
$ cd examples/privacy/diff_privacy
$ python lenet5_dp.py
```
## 3. Membership inference attack
By this attack method, we could judge whether a sample is belongs to training dataset or not.
```sh
$ cd examples/privacy/membership_inference_attack
$ python vgg_cifar_attack.py
```

