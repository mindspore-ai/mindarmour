# Application demos of model fuzzing
## Introduction
The same as the traditional software fuzz testing, we can also design fuzz test for AI models. Compared to
 branch coverage or line coverage of traditional software, some people propose the
  concept of 'neuron coverage' based on the unique structure of deep neural network. We can use the neuron coverage
   as a guide to search more metamorphic inputs to test our models.

## 1. calculation of neuron coverage 
There are three metrics proposed for evaluating the neuron coverage of a test:KMNC, NBC and SNAC. Usually we need to
 feed all the training dataset into the model first, and record the output range of all neurons (however, only the last
  layer of neurons are recorded in our method). In the testing phase, we feed test samples into the model, and
   calculate those three metrics mentioned above according to those neurons' output distribution.
```sh
$ cd examples/ai_fuzzer/
$ python lenet5_mnist_coverage.py
```
## 2. fuzz test for AI model 
We have provided several types of methods for manipulating metamorphic inputs: affine transformation, pixel
 transformation and adversarial attacks. Usually we feed the original samples into the fuzz function as seeds, and
  then metamorphic samples are generated through iterative manipulations.
```sh
$ cd examples/ai_fuzzer/
$ python lenet5_mnist_fuzzing.py
```