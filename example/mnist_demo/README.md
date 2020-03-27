# mnist demo
## Introduction

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from MNIST. The digits have been size-normalized and centered in a fixed-size image.

## run demo

### 1. download dataset
```sh
$ cd example/mnist_demo
$ mkdir MNIST_unzip
$ cd MNIST_unzip
$ mkdir train
$ mkdir test
$ cd train
$ wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
$ wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
$ gzip train-images-idx3-ubyte.gz -d
$ gzip train-labels-idx1-ubyte.gz -d
$ cd ../test
$ wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
$ wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
$ gzip t10k-images-idx3-ubyte.gz -d
$ gzip t10k-images-idx3-ubyte.gz -d
$ cd ../../
```

### 1. trian model
```sh
$ python mnist_train.py

```

### 2. run attack test
```sh
$ mkdir out.data
$ python mnist_attack_jsma.py

```

### 3. run defense/detector test
```sh
$ python mnist_defense_nad.py
$ python mnist_similarity_detector.py

```
