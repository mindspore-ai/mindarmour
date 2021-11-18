# Examples

## Introduction

This package includes application demos for all developed tools of MindArmour. Through these demos, you will soon
 master those tools of MindArmour. Let's Start!

## Preparation

Most of those demos are implemented based on LeNet5 and MNIST dataset. As a preparation, we should download MNIST and
 train a LeNet5 model first.

### 1. download dataset

The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples
. It is a subset of a larger set available from MNIST. The digits have been size-normalized and centered in a fixed-size image.

```sh
cd examples/common/dataset
mkdir MNIST
cd MNIST
mkdir train
mkdir test
cd train
wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
gzip train-images-idx3-ubyte.gz -d
gzip train-labels-idx1-ubyte.gz -d
cd ../test
wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
gzip t10k-images-idx3-ubyte.gz -d
gzip t10k-labels-idx1-ubyte.gz -d
```

### 2. trian LeNet5 model

After training the network, you will obtain a group of ckpt files. Those ckpt files save the trained model parameters
 of LeNet5, which can be used in 'examples/ai_fuzzer' and 'examples/model_security'.

```sh
cd examples/common/networks/lenet5
python mnist_train.py

```
