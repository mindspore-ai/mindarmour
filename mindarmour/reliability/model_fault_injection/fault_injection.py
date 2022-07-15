# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fault injection module
"""

import random
import numpy as np

import mindspore
from mindspore import ops, Tensor

from mindarmour.reliability.model_fault_injection.fault_type import FaultType
from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_int_positive, check_param_type, _check_array_not_empty

LOGGER = LogUtil.get_instance()
TAG = 'FaultInjector'


class FaultInjector:
    """
    Fault injection module simulates various fault scenarios for deep neural networks and evaluates
    performance and reliability of the model.

    For details, please check `Implementing the Model Fault Injection and Evaluation
    <https://mindspore.cn/mindarmour/docs/en/r1.8/fault_injection.html>`_.

    Args:
        model (Model): The model need to be evaluated.
        fi_type (list): The type of the fault injection which include bitflips_random(flip randomly),
            bitflips_designated(flip the key bit), random, zeros, nan, inf, anti_activation precision_loss etc.
        fi_mode (list): The mode of fault injection. Fault inject on just single layer or all layers.
        fi_size (list): The number of fault injection.It mean that how many values need to be injected.

    Examples:
        >>> from mindspore import Model
        >>> import mindspore.ops.operations as P
        >>> from mindarmour.reliability.model_fault_injection.fault_injection import FaultInjector
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self._softmax = P.Softmax()
        ...         self._Dense = nn.Dense(10,10)
        ...         self._squeeze = P.Squeeze(1)
        ...     def construct(self, inputs):
        ...         out = self._softmax(inputs)
        ...         out = self._Dense(out)
        ...         return self._squeeze(out)
        >>> def dataset_generator():
        ...     batch_size = 16
        ...     batches = 1
        ...     data =  np.random.randn(batches * batch_size,1,10).astype(np.float32)
        ...     label =  np.random.randint(0,10, batches * batch_size).astype(np.int32)
        ...     for i in range(batches):
        ...         yield data[i*batch_size:(i+1)*batch_size], label[i*batch_size:(i+1)*batch_size]
        >>> net = Net()
        >>> model = Model(net)
        >>> ds_eval = ds.GeneratorDataset(dataset_generator, ['image', 'label'])
        >>> fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros',
        ...            'nan', 'inf', 'anti_activation', 'precision_loss']
        >>> fi_mode = ['single_layer', 'all_layer']
        >>> fi_size = [1]
        >>> fi = FaultInjector(model, ds_eval, fi_type, fi_mode, fi_size)
        >>> fi.kick_off()
        >>> fi.metrics()
    """

    def __init__(self, model, fi_type=None, fi_mode=None, fi_size=None):
        """FaultInjector initiated."""
        self.running_list = []
        self.fi_type_map = {}
        self._init_running_list(fi_type, fi_mode, fi_size)
        self.model = model
        self._fault_type = FaultType()
        self._check_param()
        self.result_list = []
        self.original_acc = 0
        self.original_parameter = {}
        self.argmax = ops.Argmax()
        self._reducesum = ops.ReduceSum(keep_dims=False)
        self._frozen()

    def _check_param(self):
        """Check input parameters."""
        ori_attr = self._fault_type.__dir__()
        attr = []
        for attr_ in ori_attr:
            if not attr_.startswith('__') and attr_ not in ['_bitflip', '_fault_inject']:
                attr.append(attr_)
        if not isinstance(self.model, mindspore.Model):
            msg = "'Input model should be Mindspore Model', got {}.".format(type(self.model))
            LOGGER.error(TAG, msg)
            raise TypeError(msg)
        for param in self.running_list:
            if param['fi_type'] not in attr:
                msg = "'Undefined fault type', got {}.".format(self.fi_type_map[param['fi_type']])
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            if param['fi_mode'] not in ['single_layer', 'all_layer']:
                msg = "'fault mode should be single_layer or all_layer', but got {}.".format(param['fi_mode'])
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            _ = check_int_positive('fi_size', param['fi_size'])

    def _init_running_list(self, type_, mode_, size_):
        """Initiate fault injection parameters of this evaluation."""
        if type_ is None:
            type_ = ['bitflips_random', 'bitflips_designated', 'random', 'zeros', 'nan', 'inf',
                     'anti_activation', 'precision_loss']
        if mode_ is None:
            mode_ = ['single_layer', 'all_layer']
        if size_ is None:
            size_ = list(range(1, 4))
        if not isinstance(type_, list):
            msg = "'fi_type should be list', got {}.".format(type(type_))
            LOGGER.error(TAG, msg)
            raise TypeError(msg)
        if not isinstance(mode_, list):
            msg = "'fi_mode should be list', got {}.".format(type(mode_))
            LOGGER.error(TAG, msg)
            raise TypeError(msg)
        if not isinstance(size_, list):
            msg = "'fi_size should be list', got {}.".format(type(size_))
            LOGGER.error(TAG, msg)
            raise TypeError(msg)
        for i in type_:
            if not isinstance(i, str):
                msg = "'fi_type element should be str', got {} type {}.".format(i, type(i))
                LOGGER.error(TAG, msg)
                raise TypeError(msg)
            new_i = i if i.startswith('_') else '_' + i
            self.fi_type_map[new_i] = i
            for j in mode_:
                for k in size_:
                    dict_ = {'fi_type': new_i, 'fi_mode': j, 'fi_size': k}
                    self.running_list.append(dict_)

    def _frozen(self):
        """Store original parameters of model."""
        trainable_param = self.model.predict_network.trainable_params()
        for param in trainable_param:
            np_param = param.asnumpy().copy()
            bytes_ = np_param.tobytes()
            self.original_parameter[param.name] = {}
            self.original_parameter[param.name]['datatype'] = np_param.dtype
            self.original_parameter[param.name]['shape'] = np_param.shape
            self.original_parameter[param.name]['data'] = bytes_.hex()

    def _reset_model(self):
        """Reset model with original parameters."""
        for weight in self.model.predict_network.trainable_params():
            name = weight.name
            if name in self.original_parameter.keys():
                bytes_w = bytes.fromhex(self.original_parameter[name]['data'])
                datatype_w = self.original_parameter[name]['datatype']
                shape_w = self.original_parameter[name]['shape']
                np_w = np.frombuffer(bytes_w, dtype=datatype_w).reshape(shape_w)
                weight.assign_value(Tensor.from_numpy(np_w))
            else:
                msg = "Layer name not matched, got {}.".format(name)
                LOGGER.error(TAG, msg)
                raise KeyError(msg)

    @staticmethod
    def _calculate_batch_size(num, iter_times):
        """Calculate batch size based on iter_times."""
        if num <= iter_times:
            batch_list = [1] * num
            idx_list = [0] * (num + 1)
        else:
            base_batch_size = num // iter_times
            gt_num = num - iter_times * base_batch_size
            le_num = iter_times - gt_num
            batch_list = [base_batch_size + 1] * gt_num + [base_batch_size] * le_num
            idx_list = [0] * (iter_times + 1)
        for i, _ in enumerate(batch_list):
            idx_list[i + 1] = idx_list[i] + batch_list[i]
        return idx_list

    @staticmethod
    def _check_kick_off_param(ds_data, ds_label, iter_times):
        """check input data and label."""
        _ = check_int_positive('iter_times', iter_times)
        _ = check_param_type('ds_data', ds_data, np.ndarray)
        _ = _check_array_not_empty('ds_data', ds_data)
        _ = check_param_type('ds_label', ds_label, np.ndarray)
        _ = _check_array_not_empty('ds_label', ds_label)

    def kick_off(self, ds_data, ds_label, iter_times=100):
        """
        Startup and return final results after Fault Injection.

        Args:
            ds_data(np.ndarray): Input data for testing. The evaluation is based on this data.
            ds_label(np.ndarray): The label of data, corresponding to the data.
            iter_times(int): The number of evaluations, which will determine the batch size.

        Returns:
            - list, the result of fault injection.
        """
        self._check_kick_off_param(ds_data, ds_label, iter_times)
        num = ds_data.shape[0]
        idx_list = self._calculate_batch_size(num, iter_times)
        result_list = []
        for i in range(-1, len(self.running_list)):
            arg = self.running_list[i]
            total = 0
            correct = 0
            for idx in range(len(idx_list) - 1):
                a = ds_data[idx_list[idx]:idx_list[idx + 1], ...]
                batch = Tensor.from_numpy(a)
                label = Tensor.from_numpy(ds_label[idx_list[idx]:idx_list[idx + 1], ...])
                if label.ndim == 2:
                    label = self.argmax(label)
                if i != -1:
                    self._reset_model()
                    self._layer_states(arg['fi_type'], arg['fi_mode'], arg['fi_size'])
                output = self.model.predict(batch)
                predict = self.argmax(output)
                mask = predict == label
                total += predict.size
                correct += self._reducesum(mask.astype(mindspore.float32)).asnumpy()
            acc = correct / total if total else 0
            if i == -1:
                self.original_acc = acc
                result_list.append({'original_acc': self.original_acc})
            else:
                result_list.append({'type': arg['fi_type'][1:], 'mode': arg['fi_mode'], 'size': arg['fi_size'],
                                    'acc': acc, 'SDC': self.original_acc - acc})
        self._reset_model()
        self.result_list = result_list
        return result_list

    def metrics(self):
        """
        Metrics of final result.

        Returns:
            - list, the summary of result.
        """
        result_summary = []
        single_layer_acc = []
        single_layer_sdc = []
        all_layer_acc = []
        all_layer_sdc = []
        for result in self.result_list:
            if 'mode' in result.keys():
                if result['mode'] == 'single_layer':
                    single_layer_acc.append(float(result['acc']))
                    single_layer_sdc.append(float(result['SDC']))
                else:
                    all_layer_acc.append(float(result['acc']))
                    all_layer_sdc.append(float(result['SDC']))
        s_acc = np.array(single_layer_acc)
        s_sdc = np.array(single_layer_sdc)
        a_acc = np.array(all_layer_acc)
        a_sdc = np.array(all_layer_sdc)
        if single_layer_acc:
            result_summary.append('single_layer_acc_mean:%f single_layer_acc_max:%f single_layer_acc_min:%f'
                                  % (np.mean(s_acc), np.max(s_acc), np.min(s_acc)))
            result_summary.append('single_layer_SDC_mean:%f single_layer_SDC_max:%f single_layer_SDC_min:%f'
                                  % (np.mean(s_sdc), np.max(s_sdc), np.min(s_sdc)))
        if all_layer_acc:
            result_summary.append('all_layer_acc_mean:%f all_layer_acc_max:%f all_layer_acc_min:%f'
                                  % (np.mean(a_acc), np.max(a_acc), np.min(a_acc)))
            result_summary.append('all_layer_SDC_mean:%f all_layer_SDC_max:%f all_layer_SDC_min:%f'
                                  % (np.mean(a_sdc), np.max(a_sdc), np.min(a_sdc)))
        return result_summary

    def _layer_states(self, fi_type, fi_mode, fi_size):
        """FI in layer states."""
        # Choose a random layer for injection
        if fi_mode == "single_layer":
            # Single layer fault injection mode
            random_num = [random.randint(0, len(self.model.predict_network.trainable_params()) - 1)]
        elif fi_mode == "all_layer":
            # Multiple layer fault injection mode
            random_num = list(range(len(self.model.predict_network.trainable_params()) - 1))
        else:
            msg = 'undefined fi_mode {}'.format(fi_mode)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        for n in random_num:
            # Get layer states info
            w = self.model.predict_network.trainable_params()[n]
            w_np = w.asnumpy().copy()
            elem_shape = w_np.shape
            w_np = w_np.reshape(-1)

            # fault inject
            new_w_np = self._fault_type._fault_inject(w_np, fi_type, fi_size)

            # Reshape into original dimensions and store the faulty tensor
            new_w_np = np.reshape(new_w_np, elem_shape)
            w.set_data(Tensor.from_numpy(new_w_np))
