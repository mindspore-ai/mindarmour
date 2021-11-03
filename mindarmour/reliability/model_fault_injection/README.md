# Demos of model fault injection

## Introduction

This is a demo of fault injection for Mindspore applications written in Python.

## Preparation

For the demo, we should prepare both datasets and pre-train models

### Dateset

For example:

`MINST`:Download MNIST dataset from: http://yann.lecun.com/exdb/mnist/ and extract as follows

```test
File structure:
    - data_path
        - train
            - train-images-idx3-ubyte
            - train-labels-idx1-ubyte
        - test
            - t10k-images-idx3-ubyte
            - t10k-labels-idx1-ubyte
```

`CIFAR10`:Download CIFAR10 dataset from: http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz and extract as follows

```test
File structure:
    - data_path
        - train
            - data_batch_1.bin
            - data_batch_2.bin
            - data_batch_3.bin
            - data_batch_4.bin
            - data_batch_5.bin
        - test
            - test_batch.bin
```

### CheckPoint file

Download checkpoint from: https://www.mindspore.cn/resources/hub or just trained your own checkpoint

## Configuration

There are five parameters need to set up.

```python
DATA_FILE = '../common/dataset/MNIST_Data/test'
ckpt_path = '../common/networks/checkpoint_lenet_1-10_1875.ckpt'

...

fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros', 'nan', 'inf', 'anti_activation', 'precision_loss']
fi_mode = ['single_layer', 'all_layer']
fi_size = [1, 2, 3]
```

`DATA_FILE` is the directory where you store the data.

`ckpt_path` is the directory where you store the checkpoint file.

`fi_type` :
Eight types of faults can be injected. These are `bitflips_random`, `bitflips_designated`, `random`, `zeros`, `nan`, `inf`, `anti_activation` and `precision_loss`

bitflips_random: Bits are flipped randomly in the chosen value.

bitflips_designated: Specified bit is flipped in the chosen value.

random: The chosen value are replaced with random value in the range [-1, 1]

zeros: The chosen value are replaced with zero.

nan: The chosen value are replaced with NaN.

inf: The chosen value are replaced with Inf.

anti_activation: Changing the sign of the chosen value.

precision_loss: Round the chosen value to 1 decimal place

`fi_mode` :
There are twe kinds of injection modes can be specified, `single_layer` or `all_layer`.

`fi_size` is usually the exact number of values to be injected with the specified fault. For `zeros`, `anti_activation` and `precision_loss` fault, `fi_size` is the percentage of total tensor values and varies from 0% to 100%

### Example configuration

Sample 1:

```python
fi_type = ['bitflips_random', 'random', 'zeros', 'inf']
fi_mode = ['single_layer']
fi_size = [1]
```

Sample 2:

```python
fi_type = ['bitflips_designated', 'random', 'inf', 'anti_activation', 'precision_loss']
fi_mode = ['single_layer', 'all_layer']
fi_size = [1, 2]
```

## Usage

Run the test to observe the fault injection. For example:

```bash
#!/bin/bash
cd examples/reliability/
python  model_fault_injection.py --device_target GPU --device_id 2 --model lenet
```

`device_target`
`model` is the target model need to be evaluation, choose from `lenet`, `vgg16` and `resnet`, or implement your own model.

## Result

Finally, there are three kinds of result will be return.

Sample:

```test
original_acc:0.979768
type:bitflips_random mode:single_layer size:1 acc:0.968950 SDC:0.010817
type:bitflips_random mode:single_layer size:2 acc:0.948017 SDC:0.031751
...
type:precision_loss mode:all_layer size:2 acc:0.978966 SDC:0.000801
type:precision_loss mode:all_layer size:3 acc:0.979167 SDC:0.000601
single_layer_acc_mean:0.819732 single_layer_acc_max:0.980068 single_layer_acc_min:0.192107
single_layer_SDC_mean:0.160035 single_layer_SDC_max:0.787660 single_layer_SDC_min:-0.000300
all_layer_acc_mean:0.697049 all_layer_acc_max:0.979167 all_layer_acc_min:0.089443
all_layer_acc_mean:0.282719 all_layer_acc_max:0.890325 all_layer_acc_min:0.000601
```

### Original_acc

The original accuracy of model:

```test
original_acc:0.979768
```

### Specific result of each input parameter

Each result including `type`, `mode`, `size`, `acc` and `SDC`. `type`, `mode` and `size` match along with `fi_type`, `fi_mode` and `fi_size`.

```test
type:bitflips_random mode:single_layer size:1 acc:0.968950 SDC:0.010817
type:bitflips_random mode:single_layer size:2 acc:0.948017 SDC:0.031751
...
type:precision_loss mode:all_layer size:2 acc:0.978966 SDC:0.000801
type:precision_loss mode:all_layer size:3 acc:0.979167 SDC:0.000601
```

### Summary of mode

Summary of `single_layer` or `all_layer`.

```test
single_layer_acc_mean:0.819732 single_layer_acc_max:0.980068 single_layer_acc_min:0.192107
single_layer_SDC_mean:0.160035 single_layer_SDC_max:0.787660 single_layer_SDC_min:-0.000300
all_layer_acc_mean:0.697049 all_layer_acc_max:0.979167 all_layer_acc_min:0.089443
all_layer_SDC_mean:0.282719 all_layer_SDC_max:0.890325 all_layer_SDC_min:0.000601
```