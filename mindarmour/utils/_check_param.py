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

from .logger import LogUtil

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
        msg = '{} must be {}, but got {}'.format(arg_name, valid_type, type(arg_value).__name__)
        LOGGER.error(TAG, msg)
        raise TypeError(msg)

    return arg_value


def check_param_multi_types(arg_name, arg_value, valid_types):
    """Check parameter multi types."""
    if not isinstance(arg_value, tuple(valid_types)):
        msg = 'type of {} must be in {}, but got {}'.format(arg_name, valid_types, type(arg_value).__name__)
        LOGGER.error(TAG, msg)
        raise TypeError(msg)

    return arg_value


def check_int_positive(arg_name, arg_value):
    """Check positive integer."""
    # 'True' is treated as int(1) in python, which is a bug.
    if isinstance(arg_value, bool):
        msg = '{} should not be bool value, but got {}'.format(arg_name, arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    arg_value = check_param_type(arg_name, arg_value, int)
    if arg_value <= 0:
        msg = '{} must be greater than 0, but got {}'.format(arg_name, arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return arg_value


def check_value_non_negative(arg_name, arg_value):
    """Check non negative value."""
    arg_value = check_param_multi_types(arg_name, arg_value, (int, float))
    if float(arg_value) < 0.0:
        msg = '{} must not be less than 0, but got {}'.format(arg_name, arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return arg_value


def check_value_positive(arg_name, arg_value):
    """Check positive value."""
    arg_value = check_param_multi_types(arg_name, arg_value, (int, float))
    if float(arg_value) <= 0.0:
        msg = '{} must be greater than zero, but got {}'.format(arg_name, arg_value)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return arg_value


def check_param_in_range(arg_name, arg_value, lower, upper):
    """
    Check range of parameter.
    """
    if arg_value <= lower or arg_value >= upper:
        msg = '{} must be between {} and {}, but got {}'.format(arg_name, lower, upper, arg_value)
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
    msg = '{} should be an instance of {}, but got {}'.format(model_name, model_type, type(model).__name__)
    LOGGER.error(TAG, msg)
    raise TypeError(msg)


def check_numpy_param(arg_name, arg_value):
    """
    None-check and Numpy-check for `value` .

    Args:
        arg_name (str): Name of parameter.
        arg_value (numpy.ndarray): Value for check.

    Returns:
        numpy.ndarray, if `value` is not empty, return `value` with type of
        numpy.ndarray.

    Raises:
        ValueError: If value is empty.
        ValueError: If value type is not numpy.ndarray.
    """
    _ = _check_array_not_empty(arg_name, arg_value)
    if isinstance(arg_value, np.ndarray):
        arg_value = np.copy(arg_value)
    else:
        msg = 'type of {} must be numpy.ndarray, but got {}'.format(
            arg_name, type(arg_value))
        LOGGER.error(TAG, msg)
        raise TypeError(msg)
    return arg_value


def check_pair_numpy_param(inputs_name, inputs, labels_name, labels):
    """
    Dimension-equivalence check for `inputs` and `labels`.

    Args:
        inputs_name (str): Name of inputs.
        inputs (numpy.ndarray): Inputs.
        labels_name (str): Name of labels.
        labels (numpy.ndarray): Labels of `inputs`.

    Returns:
        - numpy.ndarray, if `inputs` 's dimension equals to `labels`, return inputs with type of numpy.ndarray.

        - numpy.ndarray, if `inputs` 's dimension equals to `labels` , return labels with type of numpy.ndarray.

    Raises:
        ValueError: If inputs.shape[0] is not equal to labels.shape[0].
    """
    inputs = check_numpy_param(inputs_name, inputs)
    labels = check_numpy_param(labels_name, labels)
    if inputs.shape[0] != labels.shape[0]:
        msg = '{} shape[0] must equal {} shape[0], bot got shape of ' \
              'inputs {}, shape of labels {}'.format(inputs_name, labels_name, inputs.shape, labels.shape)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return inputs, labels


def check_equal_length(para_name1, value1, para_name2, value2):
    """Check weather the two parameters have equal length."""
    if len(value1) != len(value2):
        msg = 'The dimension of {0} must equal to the {1}, but got {0} is {2}, {1} is {3}'\
            .format(para_name1, para_name2, len(value1), len(value2))
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return value1, value2


def check_equal_shape(para_name1, value1, para_name2, value2):
    """Check weather the two parameters have equal shape."""
    if value1.shape != value2.shape:
        msg = 'The shape of {0} must equal to the {1}, but got {0} is {2},  {1} is {3}'.\
            format(para_name1, para_name2, value1.shape, value2.shape)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return value1, value2


def check_norm_level(norm_level):
    """Check norm_level of regularization."""
    if not isinstance(norm_level, (int, str)):
        msg = 'Type of norm_level must be in [int, str], but got {}'.format(type(norm_level))
    accept_norm = [1, 2, '1', '2', 'l1', 'l2', 'inf', 'linf', np.inf]
    if norm_level not in accept_norm:
        msg = 'norm_level must be in {}, but got {}'.format(accept_norm, norm_level)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return norm_level


def normalize_value(value, norm_level):
    """
    Normalize gradients for gradient attacks.

    Args:
        value (numpy.ndarray): Inputs.
        norm_level (Union[int, str]): Normalized level. Option: [1, 2, np.inf, '1', '2', 'inf', 'l1', 'l2']

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
        norm = np.linalg.norm(value_reshape, ord=1, axis=1, keepdims=True) + avoid_zero_div
        norm_value = value_reshape / norm
    elif norm_level in (2, '2', 'l2'):
        norm = np.linalg.norm(value_reshape, ord=2, axis=1, keepdims=True) + avoid_zero_div
        norm_value = value_reshape / norm
    elif norm_level in (np.inf, 'inf'):
        norm = np.max(abs(value_reshape), axis=1, keepdims=True) + avoid_zero_div
        norm_value = value_reshape / norm
    else:
        msg = 'Values of `norm_level` different from 1, 2 and `np.inf` are currently not supported, but got {}.' \
            .format(norm_level)
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)
    return norm_value.reshape(ori_shape)


def check_detection_inputs(inputs, labels):
    """
    Check the inputs for detection model attacks.

    Args:
        inputs (Union[numpy.ndarray, tuple]): Images and other auxiliary inputs for detection model.
        labels (tuple): Ground-truth boxes and ground-truth labels of inputs.

    Returns:
        - numpy.ndarray, images data.

        - tuple, auxiliary inputs, such as image shape.

        - numpy.ndarray, ground-truth boxes.

        - numpy.ndarray, ground-truth labels.
    """
    if isinstance(inputs, tuple):
        has_images = False
        auxiliary_inputs = tuple()
        for item in inputs:
            check_numpy_param('item', item)
            if len(item.shape) == 4:
                images = item
                has_images = True
            else:
                auxiliary_inputs += (item,)
        if not has_images:
            msg = 'Inputs should contain images whose dimension is 4.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
    else:
        check_numpy_param('inputs', inputs)
        images = inputs
        auxiliary_inputs = ()

    check_param_type('labels', labels, tuple)
    if len(labels) != 2:
        msg = 'Labels should contain two arrays (boxes-confidences array and ground-truth labels array), ' \
              'but got {} arrays.'.format(len(labels))
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    has_boxes = False
    has_labels = False
    for item in labels:
        check_numpy_param('item', item)
        if len(item.shape) == 3:
            gt_boxes = item
            has_boxes = True
        elif len(item.shape) == 2:
            gt_labels = item
            has_labels = True
    if (not has_boxes) or (not has_labels):
        msg = 'The shape of boxes array should be (N, M, 5) or (N, M, 4), and the shape of ground-truth' \
              'labels array should be (N, M). But got {} and {}.'.format(labels[0].shape, labels[1].shape)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return images, auxiliary_inputs, gt_boxes, gt_labels


def check_inputs_labels(inputs, labels):
    """Check inputs and labels is valid for white box method."""
    _ = check_param_multi_types('inputs', inputs, (tuple, np.ndarray))
    _ = check_param_multi_types('labels', labels, (tuple, np.ndarray))
    inputs_image = inputs[0] if isinstance(inputs, tuple) else inputs
    if isinstance(inputs, tuple):
        for i, inputs_item in enumerate(inputs):
            _ = check_pair_numpy_param('inputs_image', inputs_image, 'inputs[{}]'.format(i), inputs_item)
    if isinstance(labels, tuple):
        for i, labels_item in enumerate(labels):
            _ = check_pair_numpy_param('inputs', inputs_image, 'labels[{}]'.format(i), labels_item)
    else:
        _ = check_pair_numpy_param('inputs', inputs_image, 'labels', labels)
    return inputs_image, inputs, labels


def check_param_bounds(arg_name, arg_value):
    """Check bounds is valid"""
    arg_value = check_param_multi_types(arg_name, arg_value, [tuple, list])
    if len(arg_value) != 2:
        msg = 'length of {0} must be 2, but got length of {0} is {1}'.format(arg_name, len(arg_value))
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    for i, b in enumerate(arg_value):
        if not isinstance(b, (float, int)):
            msg = 'each value in {} must be int or float, but got the {}th value is {}'.format(arg_name, i, b)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
    if arg_value[0] >= arg_value[1]:
        msg = "lower boundary must be less than upper boundary, corresponding values in {} are {} and {}". \
            format(arg_name, arg_value[0], arg_value[1])
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    return arg_value
