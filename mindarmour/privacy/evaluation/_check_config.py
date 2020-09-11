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
Verify attack config
"""

import numpy as np

from mindarmour.utils._check_param import check_param_type
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()

TAG = "check_config"


def _is_positive_int(item):
    """Verify that the value is a positive integer."""
    if not isinstance(item, int):
        return False
    return item > 0

def _is_non_negative_int(item):
    """Verify that the value is a non-negative integer."""
    if not isinstance(item, int):
        return False
    return item >= 0


def _is_positive_float(item):
    """Verify that value is a positive number."""
    if not isinstance(item, (int, float)):
        return False
    return item > 0


def _is_non_negative_float(item):
    """Verify that value is a non-negative number."""
    if not isinstance(item, (int, float)):
        return False
    return item >= 0

def _is_range_0_1_float(item):
    if not isinstance(item, (int, float)):
        return False
    return 0 <= item < 1


def _is_positive_int_tuple(item):
    """Verify that the input parameter is a positive integer tuple."""
    if not isinstance(item, tuple):
        return False
    for i in item:
        if not _is_positive_int(i):
            return False
    return True


def _is_dict(item):
    """Check whether the type is dict."""
    return isinstance(item, dict)


def _is_list(item):
    """Check whether the type is list"""
    return isinstance(item, list)


def _is_str(item):
    """Check whether the type is str."""
    return isinstance(item, str)


_VALID_CONFIG_CHECKLIST = {
    "knn": {
        "n_neighbors": [_is_positive_int],
        "weights": [{"uniform", "distance"}, callable],
        "algorithm": [{"auto", "ball_tree", "kd_tree", "brute"}],
        "leaf_size": [_is_positive_int],
        "p": [_is_positive_int],
        "metric": [_is_str, callable],
        "metric_params": [_is_dict, {None}]
    },
    "lr": {
        "penalty": [{"l1", "l2", "elasticnet", "none"}],
        "dual": [{True, False}],
        "tol": [_is_positive_float],
        "C": [_is_positive_float],
        "fit_intercept": [{True, False}],
        "intercept_scaling": [_is_positive_float],
        "class_weight": [{"balanced", None}, _is_dict],
        "random_state": None,
        "solver": [{"newton-cg", "lbfgs", "liblinear", "sag", "saga"}]
    },
    "mlp": {
        "hidden_layer_sizes": [_is_positive_int_tuple],
        "activation": [{"identity", "logistic", "tanh", "relu"}],
        "solver": [{"lbfgs", "sgd", "adam"}],
        "alpha": [_is_positive_float],
        "batch_size": [{"auto"}, _is_positive_int],
        "learning_rate": [{"constant", "invscaling", "adaptive"}],
        "learning_rate_init": [_is_positive_float],
        "power_t": [_is_positive_float],
        "max_iter": [_is_positive_int],
        "shuffle": [{True, False}],
        "random_state": None,
        "tol": [_is_positive_float],
        "verbose": [{True, False}],
        "warm_start": [{True, False}],
        "momentum": [_is_positive_float],
        "nesterovs_momentum": [{True, False}],
        "early_stopping": [{True, False}],
        "validation_fraction": [_is_range_0_1_float],
        "beta_1": [_is_range_0_1_float],
        "beta_2": [_is_range_0_1_float],
        "epsilon": [_is_positive_float],
        "n_iter_no_change": [_is_positive_int],
        "max_fun": [_is_positive_int]
    },
    "rf": {
        "n_estimators": [_is_positive_int],
        "criterion": [{"gini", "entropy"}],
        "max_depth": [{None}, _is_positive_int],
        "min_samples_split": [_is_positive_float],
        "min_samples_leaf": [_is_positive_float],
        "min_weight_fraction_leaf": [_is_non_negative_float],
        "max_features": [{"auto", "sqrt", "log2", None}, _is_positive_float],
        "max_leaf_nodes": [_is_positive_int, {None}],
        "min_impurity_decrease": [_is_non_negative_float],
        "min_impurity_split": [{None}, _is_positive_float],
        "bootstrap": [{True, False}],
        "n_jobs": [_is_positive_int, {None}],
        "random_state": None,
        "verbose": [_is_non_negative_int],
        "warm_start": [{True, False}],
        "class_weight": [{"balanced", "balanced_subsample"}, _is_dict, _is_list],
        "ccp_alpha": [_is_non_negative_float],
        "max_samples": [{None}, _is_positive_int, _is_range_0_1_float]
    }
}



def _check_config(attack_config, config_checklist):
    """
    Verify that config_list is valid.
    Check_params is the valid value range of the parameter.
    """
    for config in attack_config:
        check_param_type("config", config, dict)
        if set(config.keys()) != {"params", "method"}:
            msg = "Keys of each config in attack_config must be {}," \
                "but got {}.".format({'method', 'params'}, set(config.keys()))
            LOGGER.error(TAG, msg)
            raise KeyError(msg)

        method = str.lower(config["method"])
        params = config["params"]

        if method not in config_checklist.keys():
            msg = "Method {} is not supported.".format(method)
            LOGGER.error(TAG, msg)
            raise NameError(msg)

        if not params.keys() <= config_checklist[method].keys():
            msg = "Params in method {} is not accepted, the parameters " \
                "that can be set are {}.".format(method, set(config_checklist[method].keys()))

            LOGGER.error(TAG, msg)
            raise KeyError(msg)

        for param_key in params.keys():
            param_value = params[param_key]
            candidate_values = config_checklist[method][param_key]
            check_param_type('param_value', param_value, (list, tuple, np.ndarray))

            if candidate_values is None:
                continue

            for item_value in param_value:
                flag = False
                for candidate_value in candidate_values:
                    if isinstance(candidate_value, set) and item_value in candidate_value:
                        flag = True
                        break
                    elif not isinstance(candidate_value, set) and candidate_value(item_value):
                        flag = True
                        break

                if not flag:
                    msg = "Setting of parmeter {} in method {} is invalid".format(param_key, method)
                    raise ValueError(msg)


def verify_config_params(attack_config):
    """
    External interfaces to verify attack config.
    """
    _check_config(attack_config, _VALID_CONFIG_CHECKLIST)
