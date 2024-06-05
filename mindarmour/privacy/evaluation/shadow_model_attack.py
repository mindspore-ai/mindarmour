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
Model Inversion Attack
"""
import os
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindarmour.utils._check_param import check_param_type, \
    check_int_positive
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Model Inversion attack'


class ShadowModelLoss(nn.Cell):
    """
    The Loss function for shadow model attack.
    """
    def __init__(self, network, shadow_net, loss_fn, target_layer='conv11'):
        super(ShadowModelLoss, self).__init__()
        self._network = check_param_type('network', network, nn.Cell)
        self._shadow_net = check_param_type('shadow_net', shadow_net, nn.Cell)
        self._loss_fn = check_param_type('loss_fn', loss_fn, nn.Cell)
        self._target_layer = check_param_type('target_layer', target_layer, str)
        self._network.set_train(False)

    def construct(self, inputs, targets):
        mid_output = self._shadow_net(Tensor(inputs))
        final_output = self._network.forward_from(mid_output, self._target_layer)
        loss = self._loss_fn(final_output, targets)
        return loss


class ShadowModelAttack:
    """
    Train a shadow model based on a dataset that is known to be accessible,
    and then interrogate the shadow model to extract sensitive information.

    References: [1] HE Z, ZHANG T, LEE R B. Model inversion attacks against collaborative inference.
    https://dl.acm.org/doi/10.1145/3359789.3359824

    Args:
        network (Cell): The original network.
        shadow_network (Cell): The network used to simulate the original network.
        ckpoint_path (str): The path used to save invert model parameters.
        split_layer(str): Split target layer in split learning.

    Raises:
        TypeError: If the type of `network` or `shadow_network` is not Cell.
        ValueError: If any value of `split_layer` is not in the network.

    Examples:
            >>> import mindspore.ops.operations as P
            >>> from mindspore.nn import Cell
            >>> from mindarmour.privacy.evaluation.inversion_attack import ImageInversionAttack
            >>> from mindarmour.privacy.evaluation.shadow_model_attack import ShadowModelAttack
            >>> class Net(Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self._softmax = P.Softmax()
            ...         self._reduce = P.ReduceSum()
            ...         self._squeeze = P.Squeeze(1)
            ...     def construct(self, inputs):
            ...         out = self._softmax(inputs)
            ...         out = self._reduce(out, 2)
            ...         return self._squeeze(out)
            >>> net = Net()
            >>> shadow_net = InvNet()
            >>> original_images = np.random.random((2,1,10,10)).astype(np.float32)
            >>> target_features =  np.random.random((2,10)).astype(np.float32)
            >>> shadow_model_attack = ImageInversionAttack(net,
            ...                                         shadow_net,
            ...                                         ckpoint_path='./trained_shadow_ckpt_file',
            ...                                         split_layer='conv11')
            >>> attack_config = {'epochs': 10, 'learningrate': 1e-3, 'eps': 1e-3, 'num_classes': 10,
            ...                         'apply_ams': True}
            >>> shadow_model_attack.train_shadow_model(dataset, attack_config)
            >>> inversion_attack = ImageInversionAttack(shadow_net,
            ...                                         input_shape=(1, 10, 10),
            ...                                         input_bound=(0, 1),
            ...                                         loss_weights=[1, 0.2, 5])
            >>> inversion_images = inversion_attack.generate(target_features, iters=10)
            >>> evaluate_result = inversion_attack.evaluate(original_images, inversion_images)
    """
    def __init__(self, network, shadow_network, ckpoint_path='', split_layer='conv1'):
        self._network = check_param_type('network', network, nn.Cell)
        self._shadow_network = check_param_type('shadow_network', shadow_network, nn.Cell)
        self._split_layer = check_param_type('split_layer', split_layer, str)
        self._ckpath = check_param_type('ckpoint_path', ckpoint_path, str)
        if self._ckpath == '':
            self._ckpath = './trained_shadow_ckpt_file'
            if not os.path.exists(self._ckpath):
                os.makedirs(self._ckpath)
        else:
            load_dict = ms.load_checkpoint(self._ckpath)
            ms.load_param_into_net(self._shadow_network, load_dict)

    def train_shadow_model(self, dataset, attack_config):
        """
        Train a shadow model based on a dataset that is known to be accessible,
        and then interrogate the shadow model to extract sensitive information.

        Args:
            train_dataset (Dataset): The training dataset.
            epoch_size (int): The number of epochs. Default: 1.
            attack_config (dict): The attack configuration.

                .. code-block:: python

                    attack_config =
                        {"epochs": 50, "learningrate": 1e-3, "eps": 1e-3,
                         "num_classes": 10, "apply_ams": True}

        """
        epochs = attack_config.get('epcohs', 50)
        learningrate = attack_config.get('learningrate', 1e-3)
        eps = attack_config.get('eps', 1e-3)
        num_classes = attack_config.get('num_classes', 10)
        apply_ams = attack_config.get('apply_ams', True)

        epochs = check_int_positive('epochs', epochs)
        learningrate = check_param_type('learningrate', learningrate, float)
        eps = check_param_type('eps', eps, float)
        num_classes = check_int_positive('num_classes', num_classes)
        apply_ams = check_param_type('apply_ams', apply_ams, bool)

        net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
        optim = nn.Adam(self._shadow_network.trainable_params(), learning_rate=learningrate)
        net = ShadowModelLoss(self._network, self._shadow_network, net_loss, target_layer=self._split_layer)
        net = nn.TrainOneStepCell(net, optim)
        onehot_op = nn.OneHot(depth=num_classes)
        for epoch in range(epochs):
            loss = 0
            for inputs, targets in dataset.create_tuple_iterator():
                targets = onehot_op(targets)
                loss += net(inputs, targets).asnumpy()
            LOGGER.info(TAG, "Epoch: {}, Loss: {}".format(epoch, loss))
            if epoch % 10 == 0:
                ms.save_checkpoint(self._shadow_network, os.path.join(
                    self._ckpath, './shadow_{}_{}.ckpt'.format(self._split_layer, epoch)
                ))

    def evaluate(self, dataset, num_classes=10):
        """
        Evaluate the shadow model.

        Args:
            dataset (MappableDataset): Data for evaluation.

        Returns:
            - float, average loss.
        """
        num_classes = check_int_positive('num_classes', num_classes)
        self._shadow_network.set_train(False)

        size = 0
        total_loss = 0
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
        onehot_op = nn.OneHot(depth=num_classes)
        for inputs, targets in dataset.create_tuple_iterator():
            mid_output = self._shadow_network(Tensor(inputs))
            final_output = self._network.forward_from(mid_output, self._split_layer)
            targets = onehot_op(targets)
            total_loss += loss_fn(final_output, Tensor(targets))
            size += inputs.shape[0]
        if size != 0:
            avg_loss = total_loss / size
            return avg_loss
        return 0
