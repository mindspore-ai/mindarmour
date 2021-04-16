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
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_pair_numpy_param, check_param_type

LOGGER = LogUtil.get_instance()
TAG = "Attacker"


def _attack_knn(features, labels, param_grid, n_jobs):
    """
    Train and return a KNN model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        param_grid (dict): Setting of GridSearchCV.
        n_jobs (int): Number of jobs run in parallel. -1 means using all processors,
            otherwise the value of n_jobs must be a positive integer.

    Returns:
        sklearn.model_selection.GridSearchCV, trained model.
    """
    knn_model = KNeighborsClassifier()
    knn_model = GridSearchCV(
        knn_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=0,
    )
    knn_model.fit(X=features, y=labels)
    return knn_model


def _attack_lr(features, labels, param_grid, n_jobs):
    """
    Train and return a LR model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        param_grid (dict): Setting of GridSearchCV.
        n_jobs (int): Number of jobs run in parallel. -1 means using all processors,
            otherwise the value of n_jobs must be a positive integer.

    Returns:
        sklearn.model_selection.GridSearchCV, trained model.
    """
    lr_model = LogisticRegression(C=1.0, penalty="l2", max_iter=300)
    lr_model = GridSearchCV(
        lr_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=0,
    )
    lr_model.fit(X=features, y=labels)
    return lr_model


def _attack_mlpc(features, labels, param_grid, n_jobs):
    """
    Train and return a MLPC model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        param_grid (dict): Setting of GridSearchCV.
        n_jobs (int): Number of jobs run in parallel. -1 means using all processors,
            otherwise the value of n_jobs must be a positive integer.

    Returns:
        sklearn.model_selection.GridSearchCV, trained model.
    """
    mlpc_model = MLPClassifier(random_state=1, max_iter=300)
    mlpc_model = GridSearchCV(
        mlpc_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=0,
    )
    mlpc_model.fit(features, labels)
    return mlpc_model


def _attack_rf(features, labels, random_grid, n_jobs):
    """
    Train and return a RF model.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        random_grid (dict): Setting of RandomizedSearchCV.
        n_jobs (int): Number of jobs run in parallel. -1 means using all processors,
            otherwise the value of n_jobs must be a positive integer.

    Returns:
        sklearn.model_selection.RandomizedSearchCV, trained model.
    """
    rf_model = RandomForestClassifier(max_depth=2, random_state=0)
    rf_model = RandomizedSearchCV(
        rf_model, param_distributions=random_grid, n_iter=7, cv=3, n_jobs=n_jobs,
        verbose=0,
    )
    rf_model.fit(features, labels)
    return rf_model


def get_attack_model(features, labels, config, n_jobs=-1):
    """
    Get trained attack model specify by config.

    Args:
        features (numpy.ndarray): Loss and logits characteristics of each sample.
        labels (numpy.ndarray): Labels of each sample whether belongs to training set.
        config (dict): Config of attacker, with key in ["method", "params"].
            The format is {"method": "knn", "params": {"n_neighbors": [3, 5, 7]}},
            params of each method must within the range of changeable parameters.
            Tips of params implement can be found in
            "https://scikit-learn.org/0.16/modules/generated/sklearn.grid_search.GridSearchCV.html".
        n_jobs (int): Number of jobs run in parallel. -1 means using all processors,
            otherwise the value of n_jobs must be a positive integer.

    Returns:
        sklearn.BaseEstimator, trained model specify by config["method"].

    Examples:
        >>> features = np.random.randn(10, 10)
        >>> labels = np.random.randint(0, 2, 10)
        >>> config = {"method": "knn", "params": {"n_neighbors": [3, 5, 7]}}
        >>> attack_model = get_attack_model(features, labels, config)
    """
    features, labels = check_pair_numpy_param("features", features, "labels", labels)
    config = check_param_type("config", config, dict)
    n_jobs = check_param_type("n_jobs", n_jobs, int)
    if not (n_jobs == -1 or n_jobs > 0):
        msg = "Value of n_jobs must be -1 or positive integer."
        raise ValueError(msg)

    method = str.lower(config["method"])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        if method == "knn":
            return _attack_knn(features, labels, config["params"], n_jobs)
        if method == "lr":
            return _attack_lr(features, labels, config["params"], n_jobs)
        if method == "mlp":
            return _attack_mlpc(features, labels, config["params"], n_jobs)
        if method == "rf":
            return _attack_rf(features, labels, config["params"], n_jobs)

    msg = "Method {} is not supported.".format(config["method"])
    LOGGER.error(TAG, msg)
    raise NameError(msg)
