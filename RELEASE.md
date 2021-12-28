# MindArmour 1.2.1

## MindArmour 1.2.1 Release Notes

### Major Features and Improvements

### API Change

#### Backwards Incompatible Change

##### C++ API

[Modify] ...
[Add] ...
[Delete] ...

##### Java API

[Add] ...

#### Deprecations

##### C++ API

##### Java API

### Bug fixes

* [BUGFIX] Fix a bug of PGD method
* [BUGFIX] Fix a bug of JSMA method

### Contributors

Thanks goes to these wonderful people:

Liu Liu, Zhidan Liu, Luobin Liu and Xiulang Jin.

Contributions of any kind are welcome!

# MindArmour 1.2.0

## MindArmour 1.2.0 Release Notes

### Major Features and Improvements

#### Privacy

* [STABLE]Tailored-based privacy protection technology (Pynative)
* [STABLE]Model Inversion. Reverse analysis technology of privacy information

### API Change

#### Backwards Incompatible Change

##### C++ API

[Modify] ...
[Add] ...
[Delete] ...

##### Java API

[Add] ...

#### Deprecations

##### C++ API

##### Java API

### Bug fixes

[BUGFIX] ...

### Contributors

Thanks goes to these wonderful people:

han.yin

# MindArmour 1.1.0 Release Notes

## MindArmour

### Major Features and Improvements

* [STABLE] Attack capability of the Object Detection models.
    * Some white-box adversarial attacks, such as [iterative] gradient method and DeepFool now can be applied to Object Detection models.
    * Some black-box adversarial attacks, such as PSO and Genetic Attack now can be applied to Object Detection models.

### Backwards Incompatible Change

#### Python API

#### C++ API

### Deprecations

#### Python API

#### C++ API

### New Features

#### Python API

#### C++ API

### Improvements

#### Python API

#### C++ API

### Bug fixes

#### Python API

#### C++ API

## Contributors

Thanks goes to these wonderful people:

Xiulang Jin, Zhidan Liu, Luobin Liu and Liu Liu.

Contributions of any kind are welcome!

# Release 1.0.0

## Major Features and Improvements

### Differential privacy model training

* Privacy leakage evaluation.

    * Parameter verification enhancement.
    * Support parallel computing.

### Model robustness evaluation

* Fuzzing based Adversarial Robustness testing.

    * Parameter verification enhancement.

### Other

* Api & Directory Structure
    * Adjusted the directory structure based on different features.
    * Optimize the structure of examples.

## Bugfixes

## Contributors

Thanks goes to these wonderful people:

Liu Liu, Xiulang Jin, Zhidan Liu and Luobin Liu.

Contributions of any kind are welcome!

# Release 0.7.0-beta

## Major Features and Improvements

### Differential privacy model training

* Privacy leakage evaluation.

    * Using Membership inference to evaluate the effectiveness of privacy-preserving techniques for AI.

### Model robustness evaluation

* Fuzzing based Adversarial Robustness testing.

    * Coverage-guided test set generation.

## Bugfixes

## Contributors

Thanks goes to these wonderful people:

Liu Liu, Xiulang Jin, Zhidan Liu, Luobin Liu and Huanhuan Zheng.

Contributions of any kind are welcome!

# Release 0.6.0-beta

## Major Features and Improvements

### Differential privacy model training

* Optimizers with differential privacy

    * Differential privacy model training now supports some new policies.

    * Adaptive Norm policy is supported.

    * Adaptive Noise policy with exponential decrease is supported.  

* Differential Privacy Training Monitor

    * A new monitor is supported using zCDP as its asymptotic budget estimator.

## Bugfixes

## Contributors

Thanks goes to these wonderful people:

Liu Liu, Huanhuan Zheng, XiuLang jin, Zhidan liu.

Contributions of any kind are welcome.

# Release 0.5.0-beta

## Major Features and Improvements

### Differential privacy model training

* Optimizers with differential privacy

    * Differential privacy model training now supports both Pynative mode and graph mode.

    * Graph mode is recommended for its performance.

## Bugfixes

## Contributors

Thanks goes to these wonderful people:

Liu Liu, Huanhuan Zheng, Xiulang Jin, Zhidan Liu.

Contributions of any kind are welcome!

# Release 0.3.0-alpha

## Major Features and Improvements

### Differential Privacy Model Training

Differential Privacy is coming! By using Differential-Privacy-Optimizers, one can still train a model as usual, while the trained model preserved the privacy of training dataset, satisfying the definition of
differential privacy with proper budget.

* Optimizers with Differential Privacy([PR23](https://gitee.com/mindspore/mindarmour/pulls/23), [PR24](https://gitee.com/mindspore/mindarmour/pulls/24))

    * Some common optimizers now have a differential privacy version (SGD/Adam). We are adding more.
    * Automatically and adaptively add Gaussian Noise during training to achieve Differential Privacy.
    * Automatically stop training when Differential Privacy Budget exceeds.

* Differential Privacy Monitor([PR22](https://gitee.com/mindspore/mindarmour/pulls/22))

    * Calculate overall budget consumed during training, indicating the ultimate protect effect.

## Bug fixes

## Contributors

Thanks goes to these wonderful people:
Liu Liu, Huanhuan Zheng, Zhidan Liu, Xiulang Jin
Contributions of any kind are welcome!

# Release 0.2.0-alpha

## Major Features and Improvements

* Add a white-box attack method: M-DI2-FGSM([PR14](https://gitee.com/mindspore/mindarmour/pulls/14)).
* Add three neuron coverage metrics: KMNCov, NBCov, SNACov([PR12](https://gitee.com/mindspore/mindarmour/pulls/12)).
* Add a coverage-guided fuzzing test framework for deep neural networks([PR13](https://gitee.com/mindspore/mindarmour/pulls/13)).
* Update the MNIST Lenet5 examples.
* Remove some duplicate code.

## Bug fixes

## Contributors

Thanks goes to these wonderful people:
Liu Liu, Huanhuan Zheng, Zhidan Liu, Xiulang Jin
Contributions of any kind are welcome!

# Release 0.1.0-alpha

Initial release of MindArmour.

## Major Features

* Support adversarial attack and defense on the platform of MindSpore.
* Include 13 white-box and 7 black-box attack methods.
* Provide 5 detection algorithms to detect attacking in multiple way.
* Provide adversarial training to enhance model security.
* Provide 6 evaluation metrics for attack methods and 9 evaluation metrics for defense methods.
