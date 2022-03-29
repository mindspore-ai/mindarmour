# Copyright 2021 Huawei Technologies Co., Ltd
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
Test for fault injection.
"""

import os
import pytest
import numpy as np

from mindspore import Model
import mindspore.dataset as ds
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour.utils.logger import LogUtil
from mindarmour.reliability.model_fault_injection.fault_injection import FaultInjector

from tests.ut.python.utils.mock_net import Net

LOGGER = LogUtil.get_instance()
TAG = 'Fault injection test'
LOGGER.set_level('INFO')


def dataset_generator():
    """mock training data."""
    batch_size = 32
    batches = 128
    data = np.random.random((batches * batch_size, 1, 32, 32)).astype(
        np.float32)
    label = np.random.randint(0, 10, batches * batch_size).astype(np.int32)
    for i in range(batches):
        yield data[i * batch_size:(i + 1) * batch_size], \
              label[i * batch_size:(i + 1) * batch_size]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_fault_injector():
    """
    Feature: Fault injector
    Description: Test fault injector
    Expectation: Run kick_off and metrics successfully
    """
    # load model
    cur_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ckpt_path = os.path.join(cur_path, ckpt_path)
    net = Net()
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    model = Model(net)

    ds_eval = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
    test_images = []
    test_labels = []
    for data in ds_eval.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    ds_data = np.concatenate(test_images, axis=0)
    ds_label = np.concatenate(test_labels, axis=0)
    fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros',
               'nan', 'inf', 'anti_activation', 'precision_loss']
    fi_mode = ['single_layer', 'all_layer']
    fi_size = [1]

    # Fault injection
    fi = FaultInjector(model, fi_type, fi_mode, fi_size)
    _ = fi.kick_off(ds_data, ds_label, iter_times=100)
    _ = fi.metrics()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_wrong_model():
    """
    Feature: Fault injector
    Description: Test fault injector
    Expectation: Throw TypeError exception
    """
    # load model
    cur_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ckpt_path = os.path.join(cur_path, ckpt_path)
    net = Net()
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    ds_eval = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
    test_images = []
    test_labels = []
    for data in ds_eval.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    ds_data = np.concatenate(test_images, axis=0)
    ds_label = np.concatenate(test_labels, axis=0)
    fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros',
               'nan', 'inf', 'anti_activation', 'precision_loss']
    fi_mode = ['single_layer', 'all_layer']
    fi_size = [1]

    # Fault injection
    with pytest.raises(TypeError) as exc_info:
        fi = FaultInjector(net, fi_type, fi_mode, fi_size)
        _ = fi.kick_off(ds_data, ds_label, iter_times=100)
        _ = fi.metrics()
    assert exc_info.type is TypeError


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_wrong_data():
    """
    Feature: Fault injector
    Description: Test fault injector
    Expectation: Throw TypeError exception
    """
    # load model
    cur_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ckpt_path = os.path.join(cur_path, ckpt_path)
    net = Net()
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    model = Model(net)

    ds_data = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
    ds_label = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
    fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros',
               'nan', 'inf', 'anti_activation', 'precision_loss']
    fi_mode = ['single_layer', 'all_layer']
    fi_size = [1]

    # Fault injection
    with pytest.raises(TypeError) as exc_info:
        fi = FaultInjector(model, fi_type, fi_mode, fi_size)
        _ = fi.kick_off(ds_data, ds_label, iter_times=100)
        _ = fi.metrics()
    assert exc_info.type is TypeError


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_wrong_fi_type():
    """
    Feature: Fault injector
    Description: Test fault injector
    Expectation: Throw AttributeError exception
    """
    # load model
    cur_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ckpt_path = os.path.join(cur_path, ckpt_path)
    net = Net()
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    model = Model(net)

    ds_eval = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
    test_images = []
    test_labels = []
    for data in ds_eval.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    ds_data = np.concatenate(test_images, axis=0)
    ds_label = np.concatenate(test_labels, axis=0)
    fi_type = ['bitflips_random_haha', 'bitflips_designated', 'random', 'zeros',
               'nan', 'inf', 'anti_activation', 'precision_loss']
    fi_mode = ['single_layer', 'all_layer']
    fi_size = [1]

    # Fault injection
    with pytest.raises(ValueError) as exc_info:
        fi = FaultInjector(model, fi_type, fi_mode, fi_size)
        _ = fi.kick_off(ds_data, ds_label, iter_times=100)
        _ = fi.metrics()
    assert exc_info.type is ValueError


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_wrong_fi_mode():
    """
    Feature: Fault injector
    Description: Test fault injector
    Expectation: Throw ValueError exception
    """
    # load model
    cur_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ckpt_path = os.path.join(cur_path, ckpt_path)
    net = Net()
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    model = Model(net)

    ds_eval = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
    test_images = []
    test_labels = []
    for data in ds_eval.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    ds_data = np.concatenate(test_images, axis=0)
    ds_label = np.concatenate(test_labels, axis=0)
    fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros',
               'nan', 'inf', 'anti_activation', 'precision_loss']
    fi_mode = ['single_layer_tail', 'all_layer']
    fi_size = [1]

    # Fault injection
    with pytest.raises(ValueError) as exc_info:
        fi = FaultInjector(model, fi_type, fi_mode, fi_size)
        _ = fi.kick_off(ds_data, ds_label, iter_times=100)
        _ = fi.metrics()
    assert exc_info.type is ValueError


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_wrong_fi_size():
    """
    Feature: Fault injector
    Description: Test fault injector
    Expectation: Throw ValueError exception
    """
    # load model
    cur_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ckpt_path = os.path.join(cur_path, ckpt_path)
    net = Net()
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    model = Model(net)

    ds_eval = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
    test_images = []
    test_labels = []
    for data in ds_eval.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    ds_data = np.concatenate(test_images, axis=0)
    ds_label = np.concatenate(test_labels, axis=0)

    fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros',
               'nan', 'inf', 'anti_activation', 'precision_loss']
    fi_mode = ['single_layer', 'all_layer']
    fi_size = [-1]

    # Fault injection
    with pytest.raises(ValueError) as exc_info:
        fi = FaultInjector(model, fi_type, fi_mode, fi_size)
        _ = fi.kick_off(ds_data, ds_label, iter_times=100)
        _ = fi.metrics()
    assert exc_info.type is ValueError
