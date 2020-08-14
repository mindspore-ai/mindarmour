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
Attacker of Membership Inference.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def _attack_knn(features, labels, param_grid):
    """
    Train and return a KNN model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        param_grid (dict): Setting of GridSearchCV.

    Returns:
        sklearn.neighbors.KNeighborsClassifier, trained model.
    """
    knn_model = KNeighborsClassifier()
    knn_model = GridSearchCV(
        knn_model, param_grid=param_grid, cv=3, n_jobs=1, iid=False,
        verbose=0,
    )
    knn_model.fit(X=features, y=labels)
    return knn_model


def _attack_lr(features, labels, param_grid):
    """
    Train and return a LR model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        param_grid (dict): Setting of GridSearchCV.

    Returns:
        sklearn.linear_model.LogisticRegression, trained model.
    """
    lr_model = LogisticRegression(C=1.0, penalty="l2")
    lr_model = GridSearchCV(
        lr_model, param_grid=param_grid, cv=3, n_jobs=1, iid=False,
        verbose=0,
    )
    lr_model.fit(X=features, y=labels)
    return lr_model


def _attack_mlpc(features, labels, param_grid):
    """
    Train and return a MLPC model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        param_grid (dict): Setting of GridSearchCV.

    Returns:
        sklearn.neural_network.MLPClassifier, trained model.
    """
    mlpc_model = MLPClassifier(random_state=1, max_iter=300)
    mlpc_model = GridSearchCV(
        mlpc_model, param_grid=param_grid, cv=3, n_jobs=1, iid=False,
        verbose=0,
    )
    mlpc_model.fit(features, labels)
    return mlpc_model


def _attack_rf(features, labels, random_grid):
    """
    Train and return a RF model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        random_grid (dict): Setting of RandomizedSearchCV.

    Returns:
        sklearn.ensemble.RandomForestClassifier, trained model.
    """
    rf_model = RandomForestClassifier(max_depth=2, random_state=0)
    rf_model = RandomizedSearchCV(
        rf_model, param_distributions=random_grid, n_iter=7, cv=3, n_jobs=1,
        iid=False, verbose=0,
    )
    rf_model.fit(features, labels)
    return rf_model


def get_attack_model(features, labels, config):
    """
    Get trained attack model specify by config.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        config (dict): Config of attacker, with key in ["method", "params"].

    Returns:
        sklearn.BaseEstimator, trained model specify by config["method"].
    """
    method = str.lower(config["method"])
    if method == "knn":
        return _attack_knn(features, labels, config["params"])
    if method == "LR":
        return _attack_lr(features, labels, config["params"])
    if method == "MLP":
        return _attack_mlpc(features, labels, config["params"])
    if method == "RF":
        return _attack_rf(features, labels, config["params"])
    raise ValueError("Method {} is not support.".format(config["method"]))
