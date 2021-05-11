# Content

## Concept drift Description

In predictive analytics and machine learning, the concept drift means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. This causes problems because the predictions become less accurate as time passes. Usually, concept drift is described as the change of data distribution over time.

## Method

### Model Architecture

The concept drift detection method is based on the ESN (Echo state network). ESN is a type of reservoir computer that uses a recurrent neural network with a sparsely connected hidden layer (with typically 1% connectivity). The connectivity and weights of hidden neurons are fixed and randomly assigned.
For time series concept drift detection

### Detector

For a time series, we select two adjacent time window and compare the features of the two window data to determine whether concept drift has occurred. For feature extraction, we choose to use the ESN network. The input of the ESN network is a certain window data, and the output is also the window data (like an auto-encoder). In this way, the ESN network is equivalent to a feature extractor. Features are represented by model parameters (weights and bias) of the ESN network. Finally, by comparing the difference of model parameters, we can determine whether the data has concept drift. It should be noted that the two windows are constantly sliding forward.

## Dataset

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

## Environment Requirements

- Hardware(CPU/Ascend/GPU)
    - Prepare hardware environment with CPU, Ascend or GPU processor.
- Framework
    - MindSpore
- For more information, please check the resources below：
    - MindSpore Tutorials
    - MindSpore Python API

## Quick Start

### Initialization

```python
from mindarmour.reliability.concept_drift.concept_drift_check_time_series import ConceptDriftCheckTimeSeries

concept = ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10, step=10, threshold_index=1.5,
                                      need_label=False)
```

>window_size(int): Size of a concept window, belongs to [10, 1/3*len(input data)]. If the data is periodic, usually window_size equals 2-5 periods, such as, for monthly/weekly data, the data volume of 30/7 days is a period. Default: 100.  
rolling_window(int): Smoothing window size, belongs to [1, window_size]. Default:10.  
step(int): The jump length of the sliding window, belongs to [1,window_size]. Default:10.  
threshold_index(float): The threshold index, (-∞,+∞), Default: 1.5.  
need_label(bool）: False or True. If need_label=True, concept drift labels are needed. Default: False.  

### Data

```python
import numpy as np
file = r'archive/individual_stocks_5yr/individual_stocks_5yr/AAP_data.csv'
data = np.loadtxt(file, str, delimiter=",")
data = data[1:, 2].astype('float64')  # here we choose one column or multiple columns data[1:, 2:5].
```

>data(numpy.ndarray): Input data. The shape of data could be (n,1) or (n,m).

### Drift check

```python
drift_score, threshold, concept_drift_location = concept.concept_check(data)
```

>drift_score(numpy.ndarray): The concept drift score of the example series.  
threshold(float): The threshold to judge concept drift.  
concept_drift_location(list): The location of the concept drift.

## Script Description

```python
├── mindarmour
  ├── reliability     # descriptions about GhostNet   # shell script for evaluation with CPU, GPU or Ascend
    ├──concept_drift
        ├──concept_drift.py
        ├──readme.md
```

