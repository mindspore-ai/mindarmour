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

A tool box for MindSpore users to enhance model security and trustworthiness and protect privacy data.

MindArmour model security module is designed for adversarial examples, including four submodule: adversarial examples generation, adversarial examples detection, model defense and evaluation. The architecture is shown as follow：

![mindarmour_architecture](docs/mindarmour_architecture.png)

MindArmour differential privacy module Differential-Privacy implements the differential privacy optimizer. Currently, SGD, Momentum and Adam are supported. They are differential privacy optimizers based on the Gaussian mechanism. 
This mechanism supports both non-adaptive and adaptive policy. Rényi differential privacy (RDP) and Zero-Concentrated differential privacy（ZDP） are provided to monitor differential privacy budgets. The architecture is shown as follow：

![dp_architecture](docs/differential_privacy_architecture_en.png)

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
