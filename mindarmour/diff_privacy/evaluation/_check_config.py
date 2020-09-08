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

from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()

TAG = "check_params"


def _is_positive_int(item):
    """
    Verify that the value is a positive integer.
    """
    if not isinstance(item, int) or item <= 0:
        return False
    return True


def _is_non_negative_int(item):
    """
    Verify that the value is a non-negative integer.
    """
    if not isinstance(item, int) or item < 0:
        return False
    return True


def _is_positive_float(item):
    """
    Verify that value is a positive number.
    """
    if not isinstance(item, (int, float)) or item <= 0:
        return False
    return True


def _is_non_negative_float(item):
    """
    Verify that value is a non-negative number.
    """
    if not isinstance(item, (int, float)) or item < 0:
        return False
    return True


def _is_positive_int_tuple(item):
    """
    Verify that the input parameter is a positive integer tuple.
    """
    if not isinstance(item, tuple):
        return False
    for i in item:
        if not _is_positive_int(i):
            return False
    return True


def _is_dict(item):
    """
    Check whether the type is dict.
    """
    return isinstance(item, dict)


VALID_PARAMS_DICT = {
    "knn": {
        "n_neighbors": [_is_positive_int],
        "weights": [{"uniform", "distance"}],
        "algorithm": [{"auto", "ball_tree", "kd_tree", "brute"}],
        "leaf_size": [_is_positive_int],
        "p": [_is_positive_int],
        "metric": None,
        "metric_params": None,
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
        "solver": {"lbfgs", "sgd", "adam"},
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
        "validation_fraction": [_is_positive_float],
        "beta_1": [_is_positive_float],
        "beta_2": [_is_positive_float],
        "epsilon": [_is_positive_float],
        "n_iter_no_change": [_is_positive_int],
        "max_fun": [_is_positive_int]
    },
    "rf": {
        "n_estimators": [_is_positive_int],
        "criterion": [{"gini", "entropy"}],
        "max_depth": [_is_positive_int],
        "min_samples_split": [_is_positive_float],
        "min_samples_leaf": [_is_positive_float],
        "min_weight_fraction_leaf": [_is_non_negative_float],
        "max_features": [{"auto", "sqrt", "log2", None}, _is_positive_float],
        "max_leaf_nodes": [_is_positive_int, {None}],
        "min_impurity_decrease": {_is_non_negative_float},
        "min_impurity_split": [{None}, _is_positive_float],
        "bootstrap": [{True, False}],
        "oob_scroe": [{True, False}],
        "n_jobs": [_is_positive_int, {None}],
        "random_state": None,
        "verbose": [_is_non_negative_int],
        "warm_start": [{True, False}],
        "class_weight": None,
        "ccp_alpha": [_is_non_negative_float],
        "max_samples": [_is_positive_float]
    }
}



def _check_config(config_list, check_params):
    """
    Verify that config_list is valid.
    Check_params is the valid value range of the parameter.
    """
    if not isinstance(config_list, (list, tuple)):
        msg = "Type of parameter 'config_list' must be list, but got {}.".format(type(config_list))
        LOGGER.error(TAG, msg)
        raise TypeError(msg)

    for config in config_list:
        if not isinstance(config, dict):
            msg = "Type of each config in config_list must be dict, but got {}.".format(type(config))
            LOGGER.error(TAG, msg)
            raise TypeError(msg)

        if set(config.keys()) != {"params", "method"}:
            msg = "Keys of each config in config_list must be {}," \
                "but got {}.".format({'method', 'params'}, set(config.keys()))
            LOGGER.error(TAG, msg)
            raise KeyError(msg)

        method = str.lower(config["method"])
        params = config["params"]

        if method not in check_params.keys():
            msg = "Method {} is not supported.".format(method)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if not params.keys() <= check_params[method].keys():
            msg = "Params in method {} is not accepted, the parameters " \
                "that can be set are {}.".format(method, set(check_params[method].keys()))

            LOGGER.error(TAG, msg)
            raise KeyError(msg)

        for param_key in params.keys():
            param_value = params[param_key]
            candidate_values = check_params[method][param_key]

            if not isinstance(param_value, list):
                msg = "The parameter '{}' in method '{}' setting must within the range of " \
                "changeable parameters.".format(param_key, method)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)

            if candidate_values is None:
                continue

            for item_value in param_value:
                flag = False
                for candidate_value in candidate_values:
                    if isinstance(candidate_value, set) and item_value in candidate_value:
                        flag = True
                        break
                    elif candidate_value(item_value):
                        flag = True
                        break

                if not flag:
                    msg = "Setting of parmeter {} in method {} is invalid".format(param_key, method)
                    raise ValueError(msg)


def check_config_params(config_list):
    """
    External interfaces to verify attack config.
    """
    _check_config(config_list, VALID_PARAMS_DICT)
