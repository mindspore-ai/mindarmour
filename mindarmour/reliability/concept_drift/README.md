# Concept Drift

## Concept drift Description

In predictive analytics and machine learning, the concept drift means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. This causes problems because the predictions become less accurate as time passes. Usually, concept drift is described as the change of data distribution over time.

## Method for time series

### Model Architecture

The concept drift detection method is based on the ESN (Echo state network). ESN is a type of reservoir computer that uses a recurrent neural network with a sparsely connected hidden layer (with typically 1% connectivity). The connectivity and weights of hidden neurons are fixed and randomly assigned.
For time series concept drift detection

### Detector

For time series, we select two adjacent time window and compare the features of the two window data to determine whether concept drift has occurred. For feature extraction, we choose to use the ESN network. The input of the ESN network is a certain window data, and the output is also the window data (like an auto-encoder). In this way, the ESN network is equivalent to a feature extractor. Features are represented by model parameters (weights and bias) of the ESN network. Finally, by comparing the difference of model parameters, we can determine whether the data has concept drift. It should be noted that the two windows are constantly sliding forward.

### Dataset

Download dataset https://www.kaggle.com/camnugent/sandp500.

```bash
├── archive
  ├── all_stocks_5yr.csv
  ├── getSandP.py
  ├── merge.sh
  ├── individual_stocks_5yr
        ├──__MACOSX
        ├──individual_stocks_5yr
```

Please use the data in archive/individual_stocks_5yr/individual_stocks_5yr/XX.csv.  
In each csv file, there are 'date','open','high','low','close','volume','Name' columns, please choose one column to begin your code. 'date' and 'Name' are non-data column.  

### Environment Requirements

- Hardware(CPU/Ascend/GPU)
    - Prepare hardware environment with CPU, Ascend or GPU processor.
- Framework
    - MindSpore
- For more information, please check the resources below：
    - MindSpore Tutorials
    - MindSpore Python API

### Quick Start

#### Initialization

```python
from mindarmour.reliability.concept_drift.concept_drift_check_time_series import ConceptDriftCheckTimeSeries

concept = ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10, step=10, threshold_index=1.5,
                                      need_label=False)
```

>`window_size(int)`: Size of a concept window, no less than 10. If given the input data, window_size belongs to [10, 1/3*len(input data)]. If the data is periodic, usually window_size equals 2-5 periods, such as, for monthly/weekly data, the data volume of 30/7 days is a period. Default: 100.
`rolling_window(int)`: Smoothing window size, belongs to [1, window_size]. Default:10.  
`step(int)`: The jump length of the sliding window, belongs to [1,window_size]. Default:10.  
`threshold_index(float)`: The threshold index. Default: 1.5.  
`need_label(bool)`: False or True. If need_label=True, concept drift labels are needed. Default: False.  

#### Data

```python
import numpy as np
file = r'archive/individual_stocks_5yr/individual_stocks_5yr/AAP_data.csv'
data = np.loadtxt(file, str, delimiter=",")
data = data[1:, 2].astype('float64')  # here we choose one column or multiple columns data[1:, 2:5].
```

>`data(numpy.ndarray)`: Input data. The shape of data could be (n,1) or (n,m).

#### Drift check

```python
drift_score, threshold, concept_drift_location = concept.concept_check(data)
# the result is saved as pdf named 'concept_drift_check.pdf'
```

>`drift_score(numpy.ndarray)`: The concept drift score of the example series.  
`threshold(float)`: The threshold to judge concept drift.  
`concept_drift_location(list)`: The location of the concept drift.  


## Method for images

Generally, neural networks are used to process images. Therefore, we use algorithms based on neural networks to detect concept drifts of images. 
For image data, there is a special term that describes the concept drift in detail, Out-of-Distribution(`OOD`). 
Hereinafter, we will use the term `OOD` to describe concept drifts in images. As for non-drift images, we use the term In-Distribution(`ID`).
 
### Model Architecture
 
The model structure can be any neural network structure, such as DNN, CNN, and RNN. 
Here, we select LeNet and ResNet as examples.
 
### Detector
 
Firstly, obtain the features of the training data, the features are the outputs of a selected neural layer. 
Secondly, the features are clustered to obtain the clustering centers. 
Finally, the features of the testing data in the same neural network layer are obtained, and the distance between the testing data features and the clustering center is calculated. 
When the distance exceeds the threshold, the image is determined as an out-of-distribution(OOD) image.
 
### DataSet
 
We prepared two pairs of dataset for LeNet and ResNet separately. 
For LeNet, the training data is Mnist as ID data. The testing data is Mnist + Cifar10. Cifar10 is OOD data.
For ResNet, the training data is Cifar10 as ID data. The testing data is Cifar10 + ImageNet. ImageNet is OOD data.
 
 
### Environment Requirements
 
- Hardware
    - Prepare hardware environment with Ascend, CPU and GPU.
- Framework
    - MindSpore
- For more information, please check the resources below：
    - MindSpore Tutorials
    - MindSpore Python API

### Quick Start

#### Import

```python
import numpy as np
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore import Model, nn, context
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindarmour.reliability.concept_drift.concept_drift_check_images import OodDetectorFeatureCluster
```

#### Load Classification Model

For convenience, we use a pre-trained model file `checkpoint_lenet-10_1875.ckpt` 
in 'mindarmour/tests/ut/python/dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'. 

```python
ckpt_path = 'checkpoint_lenet-10_1875.ckpt'
net = LeNet5()
load_dict = load_checkpoint(ckpt_path)
load_param_into_net(net, load_dict)
model = Model(net)
```

>`ckpt_path(str)`: the model path.  


We can also use self-constructed model.  
It is important that we need to name the model layer, and get the layer outputs.  
Take LeNet as an example.
Firstly, we import `TensorSummary` module.   
Secondly, we initialize it as `self.summary = TensorSummary()`.  
Finally, we add `self.summary('name', x)` after each layer we pay attention to. Here,  `name` of each layer is given by users.  
After the above process, we can train the model and load it.  


```python
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import TensorSummary

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Wrap conv."""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")

def fc_with_initialize(input_channels, out_channels):
    """Wrap initialize method of full connection layer."""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

def weight_variable():
    """Wrap initialize variable."""
    return TruncatedNormal(0.05)

class LeNet5(nn.Cell):
    """
    Lenet network
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16*5*5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.summary = TensorSummary()

    def construct(self, x):
        """
        construct the network architecture
        Returns:
            x (tensor): network output
        """
        x = self.conv1(x)
        self.summary('1', x)

        x = self.relu(x)
        self.summary('2', x)

        x = self.max_pool2d(x)
        self.summary('3', x)

        x = self.conv2(x)
        self.summary('4', x)

        x = self.relu(x)
        self.summary('5', x)

        x = self.max_pool2d(x)
        self.summary('6', x)

        x = self.flatten(x)
        self.summary('7', x)

        x = self.fc1(x)
        self.summary('8', x)

        x = self.relu(x)
        self.summary('9', x)

        x = self.fc2(x)
        self.summary('10', x)

        x = self.relu(x)
        self.summary('11', x)

        x = self.fc3(x)
        self.summary('output', x)
        return x

```
#### Load Data

We prepare three datasets. The training dataset, that is the same as the dataset to train the Lenet. Two testing datasets, the first testing dataset is with OOD label(0 for non-ood, and 1 for ood) for finding an optimal threshold for ood detection.
The second testing dataset is for ood validation. The first testing dataset is not necessary if we would like to set threshold by ourselves

```python
ds_train = np.load('../../dataset/concept_train_lenet.npy')
ds_eval = np.load('../../dataset/concept_test_lenet1.npy')
ds_test = np.load('../../dataset/concept_test_lenet2.npy')
```

> `ds_train(numpy.ndarray)`: the train data.  
> `ds_eval(numpy.ndarray)`: the data for finding an optimal threshold. This dataset is not necessary.  
> `ds_test(numpy.ndarray)`: the test data for ood detection.  


#### OOD detector initialization

OOD detector for Lenet.


```python
# ood detector initialization
detector = OodDetectorFeatureCluster(model, ds_train, n_cluster=10, layer='output[:Tensor]')
```
> `model(Model)`: the model trained by the `ds_train`.  
> `ds_train(numpy.ndarray)`: the training data.  
> `n_cluster(int)`: the feature cluster number.  
> `layer(str)`: the name of the feature extraction layer.

In our example, we input the layer name `output[:Tensor]`, which can also be`9[:Tensor]`, `10[:Tensor]`, `11[:Tensor]` for LeNet. 


#### Optimal Threshold

This step is optional. If we have a labeled dataset, named `ds_eval`, we can use the following code to find the optimal detection threshold.

```python
# get optimal threshold with ds_eval
num = int(len(ds_eval) / 2)
label = np.concatenate((np.zeros(num), np.ones(num)), axis=0)  # ID data = 0, OOD data = 1
optimal_threshold = detector.get_optimal_threshold(label, ds_eval)
```

> `ds_eval(numpy.ndarray)`: the data for finding an optimal threshold.  
> `label(numpy.ndarray)`: the ood label of ds_eval. 0 means non-ood data, and 1 means ood data.

#### Detection result

```python
result = detector.ood_predict(optimal_threshold, ds_test)
```

> `ds_test(numpy.ndarray)`: the testing data for ood detection.  
> `optimal_threshold(float)`: the optimal threshold to judge out-of-distribution data. We can also set the threshold value by ourselves.

## Script Description

```bash
├── mindarmour
  ├── reliability
    ├──concept_drift
        ├──__init__.py
        ├──concept_drift_check_images.py
        ├──concept_drift_check_time_series.py
        ├──README.md
```

