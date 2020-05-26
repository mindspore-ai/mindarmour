# Release 0.3.0-alpha

## Major Features and Improvements

### Differential Privacy Model Training

Differential Privacy is coming! By using Differential-Privacy-Optimizers, one can still train a model as usual, while the trained model preserved the privacy of training dataset, satisfying the definition of
differential privacy with proper budget.
* Optimizers with Differential Privacy([PR23](https://gitee.com/mindspore/mindarmour/pulls/23), [PR24](https://gitee.com/mindspore/mindarmour/pulls/24))
    * Some common optimizers now have a differential privacy version (SGD/
    Adam). We are adding more.
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
- Add a white-box attack method: M-DI2-FGSM([PR14](https://gitee.com/mindspore/mindarmour/pulls/14)).
- Add three neuron coverage metrics: KMNCov, NBCov, SNACov([PR12](https://gitee.com/mindspore/mindarmour/pulls/12)).
- Add a coverage-guided fuzzing test framework for deep neural networks([PR13](https://gitee.com/mindspore/mindarmour/pulls/13)).
- Update the MNIST Lenet5 examples.
- Remove some duplicate code.

## Bug fixes
## Contributors
Thanks goes to these wonderful people:
Liu Liu, Huanhuan Zheng, Zhidan Liu, Xiulang Jin
Contributions of any kind are welcome!

# Release 0.1.0-alpha

Initial release of MindArmour.

## Major Features

- Support adversarial attack and defense on the platform of MindSpore.
- Include 13 white-box and 7 black-box attack methods.
- Provide 5 detection algorithms to detect attacking in multiple way.
- Provide adversarial training to enhance model security.
- Provide 6 evaluation metrics for attack methods and 9 evaluation metrics for defense methods.
