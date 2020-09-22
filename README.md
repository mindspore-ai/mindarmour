# MindArmour

- [What is MindArmour](#what-is-mindarmour)
- [Setting up](#setting-up-mindarmour)
- [Docs](#docs)
- [Community](#community)
- [Contributing](#contributing)
- [Release Notes](#release-notes)
- [License](#license)

[查看中文](./README_CN.md)

## What is MindArmour

MindArmour focus on security and privacy of artificial intelligence. MindArmour can be used as a tool box for MindSpore users to enhance model security and trustworthiness and protect privacy data.
MindArmour contains three module: Adversarial Robustness Module, Fuzz Testing Module, Privacy Protection and Evaluation Module.

### Adversarial Robustness Module

Adversarial robustness module is designed for evaluating the robustness of the model against adversarial examples, 
and provides model enhancement methods to enhance the model's ability to resist the adversarial attack and improve the model's robustness. 
This module includes four submodule: Adversarial Examples Generation, Adversarial Examples Detection, Model Defense and Evaluation. 

The architecture is shown as follow：

![mindarmour_architecture](docs/adversarial_robustness_en.png)

### Fuzz Testing Module

Fuzz Testing module is a security test for AI models. We introduce neuron coverage gain as a guide to fuzz testing according to the characteristics of neural networks. 
Fuzz testing is guided to generate samples in the direction of increasing neuron coverage rate, so that the input can activate more neurons and neuron values have a wider distribution range to fully test neural networks and explore different types of model output results and wrong behaviors.

The architecture is shown as follow：

![fuzzer_architecture](docs/fuzzer_architecture_en.png)

### Privacy Protection and Evaluation Module

Privacy Protection and Evaluation Module includes two modules: Differential Privacy Training Module and Privacy Leakage Evaluation Module.

#### Differential Privacy Training Module

Differential Privacy Training Module implements the differential privacy optimizer. Currently, SGD, Momentum and Adam are supported. They are differential privacy optimizers based on the Gaussian mechanism. 
This mechanism supports both non-adaptive and adaptive policy. Rényi differential privacy (RDP) and Zero-Concentrated differential privacy（ZCDP） are provided to monitor differential privacy budgets. 

The architecture is shown as follow：

![dp_architecture](docs/differential_privacy_architecture_en.png)

#### Privacy Leakage Evaluation Module

Privacy Leakage Evaluation Module is used to assess the risk of a model revealing user privacy. The privacy data security of the deep learning model is evaluated by using membership inference method to infer whether the sample belongs to training dataset.

The architecture is shown as follow：

![privacy_leakage](docs/privacy_leakage_en.png)


## Setting up MindArmour

### Dependencies

This library uses MindSpore to accelerate graph computations performed by many machine learning models. Therefore, installing MindSpore is a pre-requisite.  All other dependencies are included in `setup.py`.

### Installation

#### Installation for development

1. Download source code from Gitee.

```bash
git clone https://gitee.com/mindspore/mindarmour.git
```

2. Compile and install in MindArmour directory. 

```bash
$ cd mindarmour
$ python setup.py install
```

#### `Pip` installation

1. Download whl package from [MindSpore website](https://www.mindspore.cn/versions/en), then run the following command:

```
pip install mindarmour-{version}-cp37-cp37m-linux_{arch}.whl
```

2. Successfully installed, if there is no error message such as `No module named 'mindarmour'` when execute the following command:

```bash
python -c 'import mindarmour'
```

## Docs

Guidance on installation, tutorials, API, see our [User Documentation](https://gitee.com/mindspore/docs).

## Community

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/enQtOTcwMTIxMDI3NjM0LTNkMWM2MzI5NjIyZWU5ZWQ5M2EwMTQ5MWNiYzMxOGM4OWFhZjI4M2E5OGI2YTg3ODU1ODE2Njg1MThiNWI3YmQ) - Ask questions and find answers.

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for more details.

## Release Notes

The release notes, see our [RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
