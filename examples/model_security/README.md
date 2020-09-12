# Application demos of model security
## Introduction
It has been proved that AI models are vulnerable to adversarial noise that invisible to human eye. Through those
 demos in this package, you will learn to use the tools provided by MindArmour to generate adversarial samples and
  also improve the robustness of your model.

## 1. Generate adversarial samples (Attack method)
Attack methods can be classified into white box attack and black box attack. White-box attack means that the attacker
 is accessible to the model structure and its parameters. Black-box means that the attacker can only obtain the predict
  results of the
  target model.
### white-box attack
Running the classical attack method: FGSM-Attack.
```sh
$ cd examples/model_security/model_attacks/white-box
$ python mnist_attack_fgsm.py
```
### black-box attack
Running the classical black method: PSO-Attack.
```sh
$ cd examples/model_security/model_attacks/black-box
$ python mnist_attack_pso.py
```
## 2. Improve the robustness of models
### adversarial training
Adversarial training is an effective method to enhance the model's robustness to attacks, in which generated
 adversarial samples are fed into the model for retraining.
 ```sh
$ cd examples/model_security/model_defenses
$ python mnist_defense_nad.py
```
### adversarial detection
Besides adversarial training, there is another type of defense method: adversarial detection. This method is mainly
 for black-box attack. The reason is that black-box attacks usually require frequent queries to the model, and the
  difference between adjacent queries input is small. The detection algorithm could analyze the similarity of a series
   of queries and recognize the attack.
 ```sh
$ cd examples/model_security/model_defenses
$ python mnist_similarity_detector.py
```