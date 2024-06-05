# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This module receive and process configuration to ensure they conform to the VFL.
"""
import os
import yaml

__all__ = ["argsments_function"]

def argsments_function(epochs=100, batch_size=64, lr=0.01, top_model_trainable=1, n_party=2, adversary=1,
                       gpu=0, target_train_size=-1, target_test_size=-1, backdoor_test_size=2000, backdoor_label=0,
                       alg='no', num_classes=10, poison_num=10, topk=1):
    """
    Arguments processing.

    Args:
        epochs (int): Training epochs.
        batch_size (int): The size of each batch.
        lr (float): Learning rate.
        top_model_trainable (int): VFL with splitting or VFL without splitting.
        n_party (int): The number of parties.
        adversary (int): The ID of the attacker.
        gpu (int): Use GPU or not.
        target_train_size (int): The size of the train dataset.
        target_test_size (int): The size of the test dataset.
        backdoor_test_size (int): The size of the backdoor test dataset.
        backdoor_label (int): Backdoor target label.
        alg (str): The name of the algorithm.
        num_classes (int): The number of classes.
        poison_num (int): The number of poisoned samples.
        topk (int): The top-k metric.

    Returns:
        dict: The processed arguments.
    """
    if alg == 'direct_attack' and top_model_trainable:
        return
    if alg == 'g_r' and not top_model_trainable:
        return
    if n_party != 2:
        return
    if adversary != 1:
        return
    if backdoor_label >= num_classes:
        return
    yaml.warnings({'YAMLLoadWarning': False})
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/config.yaml', 'r', encoding='utf-8')
    cfg = f.read()
    args = yaml.load(cfg, Loader=yaml.SafeLoader)
    f.close()
    args['num_classes'] = num_classes
    args['target_epochs'] = epochs
    args['passive_bottom_lr'] = lr
    args['active_bottom_lr'] = lr
    args['active_top_lr'] = lr
    args['target_batch_size'] = batch_size
    args['passive_bottom_gamma'] = 0.1
    args['active_bottom_gamma'] = 0.1
    args['active_top_gamma'] = 0.1
    args['cuda'] = gpu
    args['active_top_trainable'] = bool(top_model_trainable)
    args['n_passive_party'] = n_party - 1
    args['adversary'] = adversary
    args['target_train_size'] = target_train_size
    args['target_test_size'] = target_test_size
    args['backdoor_test_size'] = backdoor_test_size
    args['backdoor_label'] = backdoor_label
    args['topk'] = topk
    args['aggregate'] = 'Concate'
    if alg == 'g_r':
        args['attack'] = True
        args['backdoor'] = alg
        args['label_inference_attack'] = 'no'
        args['trigger'] = 'pixel'
        args['trigger_add'] = False
    if alg == 'direct_attack':
        args['attack'] = True
        args['backdoor'] = 'no'
        args['label_inference_attack'] = alg
    args['amplify_ratio'] = 1
    args['backdoor_train_size'] = poison_num
    return args
