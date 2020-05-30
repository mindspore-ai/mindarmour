# Copyright 2019 Huawei Technologies Co., Ltd
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
""" check parameters for MindArmour. """
import numpy as np

from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'check parameters'


def _check_array_not_empty(arg_name, arg_value):
    """Check parameter is empty or not."""
    if isinstance(arg_value, (tuple, list)):
        if not arg_value:
            msg = '{} must not be empty'.format(arg_name)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

    if isinstance(arg_value, np.ndarray):
        if arg_value.size <= 0:
            msg = '{} must not be empty'.format(arg_name)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
    return arg_value


def check_param_type(arg_name, arg_value, valid_type):
    """Check parameter type."""
    if not isinstance(arg_value, valid_type):
        msg = '{} must be {}, but got {}'.format(arg_name,
                                                 valid_type,
                                                 type(arg_value).__name__)
        LOGGER.error(TAG, msg)
        raise TypeError(msg)

    return arg_value


def check_param_multi_types(arg_name, arg_value, valid_types):
    """Check parameter type."""
    if not isinstance(arg_value, tuple(valid_types)):
        msg = 'type of {} must be in {}, but got {}' \
            .format(arg_name, valid_types, type(arg_value).__name__)
        LOGGER.error(TAG, msg)
        raise TypeError(msg)

    return arg_value


def check_int_positive(arg_name, arg_value):
    """Check positive integer."""
    arg_value = check_param_type(arg_name, arg_value, int)
    if arg_value <= 0:
        msg = '{} must be greater than 0, but got {}'.format(arg_name,
                                                             arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return arg_value


def check_value_non_negative(arg_name, arg_value):
    """Check non negative value."""
    arg_value = check_param_multi_types(arg_name, arg_value, (int, float))
    if float(arg_value) < 0.0:
        msg = '{} must not be less than 0, but got {}'.format(arg_name,
                                                              arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return arg_value


def check_value_positive(arg_name, arg_value):
    """Check positive value."""
    arg_value = check_param_multi_types(arg_name, arg_value, (int, float))
    if float(arg_value) <= 0.0:
        msg = '{} must be greater than zero, but got {}'.format(arg_name,
                                                                arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return arg_value


def check_param_in_range(arg_name, arg_value, lower, upper):
    """
    Check range of parameter.
    """
    if arg_value <= lower or arg_value >= upper:
        msg = '{} must be between {} and {}, but got {}'.format(arg_name,
                                                                lower,
                                                                upper,
                                                                arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)

    return arg_value


def check_model(model_name, model, model_type):
    """
    Check the type of input `model` .

    Args:
        model_name (str): Name of model.
        model (Object): Model object.
        model_type (Class): Class of model.

    Returns:
        Object, if the type of `model` is `model_type`, return `model` itself.

    Raises:
        ValueError: If model is not an instance of `model_type` .
    """
    if isinstance(model, model_type):
        return model
    msg = '{} should be an instance of {}, but got {}' \
        .format(model_name,
                model_type,
                type(model).__name__)
    LOGGER.error(TAG, msg)
    raise ValueError(msg)


def check_numpy_param(arg_name, arg_value):
    """
    None-check and Numpy-check for `value` .

    Args:
        arg_name (str): Name of parameter.
        arg_value (Union[list, tuple, numpy.ndarray]): Value for check.

    Returns:
        numpy.ndarray, if `value` is not empty, return `value` with type of
        numpy.ndarray.

    Raises:
        ValueError: If value is empty.
        ValueError: If value type is not in (list, tuple, numpy.ndarray).
    """
    _ = _check_array_not_empty(arg_name, arg_value)
    if isinstance(arg_value, (list, tuple)):
        arg_value = np.asarray(arg_value)
    elif isinstance(arg_value, np.ndarray):
        arg_value = np.copy(arg_value)
    else:
        msg = 'type of {} must be in (list, tuple, numpy.ndarray)'.format(
            arg_name)
        LOGGER.error(TAG, msg)
        raise TypeError(msg)
    return arg_value


def check_pair_numpy_param(inputs_name, inputs, labels_name, labels):
    """
    Dimension-equivalence check for `inputs` and `labels`.

    Args:
        inputs_name (str): Name of inputs.
        inputs (Union[list, tuple, numpy.ndarray]): Inputs.
        labels_name (str): Name of labels.
        labels (Union[list, tuple, numpy.ndarray]): Labels of `inputs`.

    Returns:
        - Union[list, tuple, numpy.ndarray], if `inputs` 's dimension equals to
          `labels`, return inputs with type of numpy.ndarray.

        - Union[list, tuple, numpy.ndarray], if `inputs` 's dimension equals to
          `labels` , return labels with type of numpy.ndarray.

    Raises:
        ValueError: If inputs.shape[0] is not equal to labels.shape[0].
    """
    inputs = check_numpy_param(inputs_name, inputs)
    labels = check_numpy_param(labels_name, labels)
    if inputs.shape[0] != labels.shape[0]:
        msg = '{} shape[0] must equal {} shape[0], bot got shape of ' \
              'inputs {}, shape of labels {}'.format(inputs_name, labels_name,
                                                     inputs.shape, labels.shape)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return inputs, labels


def check_equal_length(para_name1, value1, para_name2, value2):
    """check weather the two parameters have equal length."""
    if len(value1) != len(value2):
        msg = 'The dimension of {0} must equal to the ' \
              '{1}, but got {0} is {2}, ' \
              '{1} is {3}'.format(para_name1, para_name2, len(value1),
                                  len(value2))
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return value1, value2


def check_equal_shape(para_name1, value1, para_name2, value2):
    """check weather the two parameters have equal shape."""
    if value1.shape != value2.shape:
        msg = 'The shape of {0} must equal to the ' \
              '{1}, but got {0} is {2}, ' \
              '{1} is {3}'.format(para_name1, para_name2, value1.shape[0],
                                  value2.shape[0])
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return value1, value2


def check_norm_level(norm_level):
    """
    check norm_level of regularization.
    """
    accept_norm = [1, 2, '1', '2', 'l1', 'l2', 'inf', 'linf', np.inf]
    if norm_level not in accept_norm:
        msg = 'norm_level must be in {}, but got {}'.format(accept_norm,
                                                            norm_level)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return norm_level


def normalize_value(value, norm_level):
    """
    Normalize gradients for gradient attacks.

    Args:
        value (numpy.ndarray): Inputs.
        norm_level (Union[int, str]): Normalized level.

    Returns:
        numpy.ndarray, normalized value.

    Raises:
        NotImplementedError: If norm_level is not in [1, 2 , np.inf, '1', '2',
            'inf', 'l1', 'l2']
    """
    norm_level = check_norm_level(norm_level)
    ori_shape = value.shape
    value_reshape = value.reshape((value.shape[0], -1))
    avoid_zero_div = 1e-12
    if norm_level in (1, '1', 'l1'):
        norm = np.linalg.norm(value_reshape, ord=1, axis=1, keepdims=True) + \
               avoid_zero_div
        norm_value = value_reshape / norm
    elif norm_level in (2, '2', 'l2'):
        norm = np.linalg.norm(value_reshape, ord=2, axis=1, keepdims=True) + \
               avoid_zero_div
        norm_value = value_reshape / norm
    elif norm_level in (np.inf, 'inf'):
        norm = np.max(abs(value_reshape), axis=1, keepdims=True) + \
               avoid_zero_div
        norm_value = value_reshape / norm
    else:
        msg = 'Values of `norm_level` different from 1, 2 and ' \
              '`np.inf` are currently not supported, but got {}.' \
            .format(norm_level)
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)
    return norm_value.reshape(ori_shape)
