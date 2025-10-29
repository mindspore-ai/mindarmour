# MindBackdoor

A MindSpore-based library for benchmarking **backdoor attacks and defenses** in deep learning.  
It provides clean implementations of popular poisoning attacks and defenses, plus modular dataset/model wrappers for fine-grained control and reproducible experiments.

## Features

- **Multiple attack and defense algorithms:** Covering adaptive, blended, physical, trigger-based, feature-space, and other advanced backdoor strategies.
- **Flexible dataset handling:** `DatasetFolder` and `Compose` (torch-style) for easy sample-level transformation and composition.
- **Layer-wise feature extraction:** Model outputs are annotated at each layer for attack/defense feature access.
- **MindSpore first:** All implementations are based on [MindSpore](https://www.mindspore.cn/), easy to run on both CPU and GPU.

## Supported Datasets

- **CIFAR-10** (recommended, default for all examples, and for general scenarios)
- **GTRSB** (for traffic scenario)
- **MedMNIST** (for medicine scenario)
- **Flower102** (for traffic scenario)
- **FashionMNIST** (for advertisement scenario)
- Any dataset following the `DatasetFolder` image structure

### DatasetFolder Image Structure

`DatasetFolder` supports any dataset organized as an image folder, similar to the [PyTorch ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) style.  
**Each class should have its own subdirectory, and images should be placed inside their respective class folders.**

**Example:**

```
/path/to/your-dataset/   
├── class1/    
│   ├── xxx1.png    
│   ├── xxx2.png    
│   └── ...    
├── class2/    
│   ├── xxy1.png    
│   ├── xxy2.png   
│   └── ...    
└── class3/    
    ├── xxz1.png    
    ├── xxz2.png    
    └── ...   
```

- The class name (folder name) will be automatically used as the label and objected to digit label (e.g. 0, 1, 2, ...).
- The data loader will recursively read images in each subdirectory.

**To use a custom dataset:**
- Arrange your images into the above structure, with one folder per class.
- Specify the root directory when initializing `DatasetFolder`.

**Sample code:**
```python
trainset = DatasetFolder(
    root='/path/to/your-dataset/train',
    transform=transform_train,
)
testset = DatasetFolder(
    root='/path/to/your-dataset/test',
    transform=transform_test,
)
```

**Note:**

+ This structure works for any classification task, and makes it easy to use new datasets with minimal modification.

### Data Preprocessing Convention

For **all datasets**, regardless of their original size or number of channels, we standardize the data before training and evaluation:

- **All images are automatically resized to 32×32 pixels.**
- **All images are converted to 3 channels (RGB) if not already.**

This normalization ensures that all models and attacks/defenses are compatible and results are comparable across different scenarios.

**How it works:**
- The included `transform_train` and `transform_test` pipelines automatically handle resizing and channel conversion.
- You can further customize preprocessing by editing the `Compose` transform list.

**Example:**
```python
from mindspore.dataset.vision import Resize, ToTensor, RandomHorizontalFlip, Grayscale

transform_train = Compose([
    # Grayscale(num_output_channels=3), # Converts image to 3-channel grayscale (for grayscale datasets)
    Resize((32, 32)),       # Ensures output is 32x32
    ToTensor(),             # Converts to (C, H, W) and normalizes
    RandomHorizontalFlip(), # Data augmentation (optional)
    # ... add more transforms if needed
])
```

**Note:**

+ Even grayscale or single-channel datasets (e.g., FashionMNIST) will be converted to three channels by `Grayscale(num_output_channels=3)`.
+ If your dataset has images of different sizes or color formats, `Resize((32, 32))` will automatically adapt them to the expected input format.


## Supported Attacks

- `AdaptivePatch`
- `AdaptiveBlend`
- `AdaptiveKWay`
- `Blended`
- `PhysicalBA`
- `BATT`
- `WaNet`
- `TUAP`
- (and maybe more...)

## Supported Defenses

- `AutoEncoderDefense`
- `Pruning`
- `FineTuning`
- `Spectral`
- `ShrinkPad`
- `Beatrix`
- (and maybe more...)

---

## Installation

> **Requirements:**  
> - MindSpore >= 2.0 (GPU or CPU version)  
> - Python 3.8+

You may need to follow [MindSpore](https://www.mindspore.cn/)’s official installation guide for your platform.

## Quick Start

### **Example: Running a Blended Attack**

```python
from core.attacks import Blended
from core.dataset import DatasetFolder, Compose
from core.models import ResNet18
from mindspore import nn
from mindspore.dataset.vision import Resize, ToTensor, RandomHorizontalFlip

import numpy as np

# 1. Data transforms and dataset
transform_train = Compose([
    Resize((32, 32)),
    ToTensor(),
    RandomHorizontalFlip(),
])
transform_test = Compose([
    Resize((32, 32)),
    ToTensor()
])
trainset = DatasetFolder(root='/path/to/cifar10/train', transform=transform_train)
testset = DatasetFolder(root='/path/to/cifar10/test', transform=transform_test)

# 2. Pattern and mask
pattern = np.zeros((1, 32, 32), dtype=np.uint8)
pattern[0, -3:, -3:] = 255
weight = np.zeros((1, 32, 32), dtype=np.float32)
weight[0, -3:, -3:] = 0.2

# 3. Model and attacker
model = ResNet18(num_classes=10)
attacker = Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(reduction='mean'),
    y_target=0,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=1,
    poisoned_transform_test_index=1,
    poisoned_target_transform_index=0,
)
schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': '0',
    'benign_training': False,
    'batch_size': 128,
    'num_workers': 1,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'warmup_epoch': 5,
    'decay_epoch': 200,
    'epochs': 200,
    'log_iteration_interval': 100,
    'test_epoch_interval': 5,
}
attacker.train(schedule)
```

### **Example: Running a AutoEncoder Defense**

```python
"""
Example: Backdoor Attack and Defense Workflow (MindBackdoor)
"""

from core.attacks import Blended
from core.defenses import AutoEncoderDefense
from core.dataset import DatasetFolder, Compose
from core.models import ResNet18
from mindspore import nn, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.dataset.vision import Resize, ToTensor, RandomHorizontalFlip, Grayscale

import numpy as np

# Step 1: Define data transforms
transform_train = Compose([
    Grayscale(num_output_channels=3),  # Converts input to 3-channel grayscale
    Resize((32, 32)),                  # Ensures all images are 32x32
    ToTensor(),                        # Converts to (C, H, W) float32, range [0, 1]
    RandomHorizontalFlip(),            # Data augmentation (optional)
])

transform_test = Compose([
    Grayscale(num_output_channels=3),
    Resize((32, 32)),
    ToTensor(),
])

# Step 2: Construct dataset
trainset = DatasetFolder(
    root='/path/to/your/train',
    transform=transform_train,
)

testset = DatasetFolder(
    root='/path/to/your/test',
    transform=transform_test,
)

# Step 3: Prepare attack pattern and weight
pattern = np.zeros((1, 32, 32), dtype=np.uint8)
pattern[0, -3:, -3:] = 255
weight = np.zeros((1, 32, 32), dtype=np.float32)
weight[0, -3:, -3:] = 0.2

# Step 4: Model definition
model = ResNet18(num_classes=10)

# Step 5: Attack object
attacker = Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(reduction='mean'),
    y_target=0,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=1, # after Resize((32, 32))
    poisoned_transform_test_index=1,
    poisoned_target_transform_index=0,
)

schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': '0',
    'benign_training': False,
    'batch_size': 128,
    'num_workers': 1,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'warmup_epoch': 5,
    'decay_epoch': 200,
    'epochs': 200,
    'log_iteration_interval': 100,
    'test_epoch_interval': 5,
}

# Step 6: Train attacked model
attacker.train(schedule)
print('Attack done')

# Step 7: Save/load backdoored model (optional)
bd_model = model
# save_checkpoint(bd_model, "./temp/bd_model.ckpt")
# bd_model = ResNet18(num_classes=10)
# param_dict = load_checkpoint("./temp/bd_model.ckpt")
# load_param_into_net(bd_model, param_dict)

# Step 8: Get poisoned datasets
poisoned_trainset, poisoned_testset = attacker.get_poisoned_dataset()

# Step 9: Defense (AutoEncoder)
print('Start training AutoEncoder')
defense = AutoEncoderDefense(
    train_dataset=trainset,
    test_dataset=testset,
    pretrain_path=None # or "./temp/autoencoder.ckpt"
)

# save_checkpoint(defense.get_autoencoder(), "./temp/autoencoder.ckpt")

print('Start testing AutoEncoder Defense')
defense.test(
    model=bd_model,
    dataset=poisoned_testset,
)
```

## How to Extend

- **Add new attacks/defenses:** Inherit from `Base` in `core.attacks`/`core.defenses` and refer to the structure of the provided methods.
- **Use your own dataset:** Make sure it follows the DatasetFolder interface..
- **Use other MindSpore models:** Just ensure layer outputs are accessible for your algorithm for some defense methods.

## FAQ

- **Does it support PyTorch?**
  No. This library is MindSpore-only for best compatibility and performance in the MindSpore ecosystem. You could refer to this [repo](https://github.com/THUYimingLi/BackdoorBox) for Pytorch implementation.

- **How are models structured?**
  Each model exposes layer-wise outputs (e.g., layer4) for feature-based detection methods.

- **How to customize transformations?**
  You can compose any MindSpore vision transforms using the `Compose` class and follow the guide in [Jump to Supported Datasets](#supported-datasets)

## Contributors

+ [ZJU NESA Lab](https://nesa.zju.edu.cn/)
