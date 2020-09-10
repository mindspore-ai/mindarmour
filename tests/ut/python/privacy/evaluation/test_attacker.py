# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
attacker test
"""
import pytest

import numpy as np

from mindarmour.privacy.evaluation.attacker import get_attack_model


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_get_knn_model():
    features = np.random.randint(0, 10, [100, 10])
    labels = np.random.randint(0, 2, [100])
    config_knn = {
        "method": "KNN",
        "params": {
            "n_neighbors": [3, 5, 7],
        }
    }
    knn_attacker = get_attack_model(features, labels, config_knn, -1)
    pred = knn_attacker.predict(features)
    assert pred is not None


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_get_lr_model():
    features = np.random.randint(0, 10, [100, 10])
    labels = np.random.randint(0, 2, [100])
    config_lr = {
        "method": "LR",
        "params": {
            "C": np.logspace(-4, 2, 10),
        }
    }
    lr_attacker = get_attack_model(features, labels, config_lr, -1)
    pred = lr_attacker.predict(features)
    assert pred is not None


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_get_mlp_model():
    features = np.random.randint(0, 10, [100, 10])
    labels = np.random.randint(0, 2, [100])
    config_mlpc = {
        "method": "MLP",
        "params": {
            "hidden_layer_sizes": [(64,), (32, 32)],
            "solver": ["adam"],
            "alpha": [0.0001, 0.001, 0.01],
        }
    }
    mlpc_attacker = get_attack_model(features, labels, config_mlpc, -1)
    pred = mlpc_attacker.predict(features)
    assert pred is not None


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_get_rf_model():
    features = np.random.randint(0, 10, [100, 10])
    labels = np.random.randint(0, 2, [100])
    config_rf = {
        "method": "RF",
        "params": {
            "n_estimators": [100],
            "max_features": ["auto", "sqrt"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    }
    rf_attacker = get_attack_model(features, labels, config_rf, -1)
    pred = rf_attacker.predict(features)
    assert pred is not None
