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

from mindarmour.utils._check_param import check_param_type, check_int_positive
from mindarmour.utils.logger import LogUtil
from mindarmour.utils.util import compute_ssim, compute_psnr

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Model Inversion attack'


class ModelInversionLoss(nn.Cell):
    """
    The loss function for model inversion attack.

    Args:
        network (Cell): The network used to infer images' intermediate representations.
        invnet (Cell): The network used to reply to the original image input through intermediate representations.
        loss_fn (Cell): The Loss function used to calculate the distance between the original input and the reply input.
        target_layer (str): Split target layer in split learning.
    """
    def __init__(self, network, invnet, loss_fn, target_layer='conv1'):
        super(ModelInversionLoss, self).__init__()
        self._network = check_param_type('network', network, nn.Cell)
        self._invnet = check_param_type('invnet', invnet, nn.Cell)
        self._loss_fn = check_param_type('loss_fn', loss_fn, nn.Cell)
        self._target_layer = check_param_type('target_layer', target_layer, str)
        self._network.set_train(False)

    def construct(self, inputs):
        orginal_model_output = self._network.getLayerOutput(inputs, self._target_layer)
        decoder_model_output = self._invnet(orginal_model_output)
        loss = self._loss_fn(inputs, decoder_model_output)
        return loss


class ModelInversionAttack:
    """
    An model attack method used to reconstruct images by invert their model.

    References: [1] HE Z, ZHANG T, LEE R B. Model inversion attacks against collaborative inference.
    https://dl.acm.org/doi/10.1145/3359789.3359824

    Args:
        network (Cell): The network used to infer images' intermediate representations.
        ckpoint_path (str): The path used to save invert model parameters.
        split_layer(str): Split target layer in split learning.
        ds_name (str): The name of the dataset used to train the reverse model.

    Raises:
        TypeError: If the type of `network` is not Cell.
    """
    def __init__(self, network, inv_network, input_shape, ckpoint_path=None, split_layer='conv1'):
        self._network = check_param_type('network', network, nn.Cell)
        self._invnetwork = check_param_type('inv_network', inv_network, nn.Cell)
        self._split_layer = check_param_type('split_layer', split_layer, str)
        self._ckpath = check_param_type('ckpoint_path', ckpoint_path, str)
        self.check_inv_network(input_shape)
        if self._ckpath is None:
            self._ckpath = './trained_inv_ckpt_file'
        else:
            load_dict = ms.load_checkpoint(self._ckpath)
            ms.load_param_into_net(self._invnet, load_dict)

    def check_inv_network(self, input_shape):
        input_shape = check_param_type('input_shape', input_shape, tuple)
        inputs = ms.numpy.ones((1,) + input_shape)
        orginal_model_output = self._network.getLayerOutput(inputs, self._split_layer)
        inv_model_output = self._invnetwork(orginal_model_output)
        if inputs.shape != inv_model_output.shape:
            msg = "InvModel error, input shape is {}, but invmodel output shape is {}" \
                .format(inputs.shape, inv_model_output.shape)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

    def train_inversion_model(self, dataset, epochs=50, learningrate=1e-3, eps=1e-3, apply_ams=True):
        """
        Train reverse model based on dataset

        Args:
            dataset (MappableDataset): Data for training reverse models.
            Nepochs (int): Training rounds, which should be positive integers. Defalult: ``50``.
            LearninigRate (float): The learning rate used to update the parameters of the reverse model.
            eps (float): The epsilon used to update the parameters of the reverse model.
            AMSGrad (bool): Whether to use AMSGrad to update the parameters of the reverse model.

        Raises:
            TypeError: If the type of `dataset` is not MappableDataset.
            TypeError: If the type of `Nepochs` is not int.
            TypeError: If the type of `LearningRate` is not float.
            TypeError: If the type of `eps` is not float.
            TypeError: If the type of `AMSGrad` is not bool.
        """
        epochs = check_int_positive('epochs', epochs)
        learningrate = check_param_type('learningrate', learningrate, float)
        eps = check_param_type('eps', eps, float)
        apply_ams = check_param_type('apply_ams', apply_ams, bool)

        self._invnet.set_train(True)
        net_loss = nn.MSELoss()
        optim = nn.Adam(self._invnet.trainable_params(), learning_rate=learningrate, eps=eps, use_amsgrad=apply_ams)
        net = ModelInversionLoss(self._network, self._invnet, net_loss, self._split_layer)
        net = nn.TrainOneStepCell(net, optim)

        for epoch in range(epochs):
            loss = 0
            for inputs, _ in dataset.create_tuple_iterator():
                loss += net(Tensor(inputs)).asnumpy()
            LOGGER.info(TAG, "Epoch: {}, Loss: {}".format(epoch, loss))
            if epoch % 10 == 0:
                ms.save_checkpoint(self._invnet, os.path.join(self._ckpath, '/invmodel_{}_{}.ckpt'
                                                              .format(self._split_layer, epoch)))

    def evaluate(self, dataset):
        """
        Evaluate the model inversion attack.

        Args:
            dataset (MappableDataset): Data for evaluation.

        Returns:
            - float, average ssim value.
            - float, average psnr value.

        """
        self._invnet.set_train(False)

        total_ssim = 0
        total_psnr = 0
        size = 0
        for inputs, _ in dataset.create_tuple_iterator():
            orginal_model_output = self._network.getLayerOutput(Tensor(inputs), self._split_layer)
            decoder_model_output = self._invnet(orginal_model_output)
            decoder_model_output = decoder_model_output.clip(0, 1)
            for i in range(inputs.shape[0]):
                original_image = inputs[i].transpose(1, 2, 0).asnumpy()
                compared_image = decoder_model_output[i].transpose(1, 2, 0).asnumpy()
                ssim = compute_ssim(original_image, compared_image)
                psnr = compute_psnr(original_image, compared_image)

                total_ssim += ssim
                total_psnr += psnr
            size += inputs.shape[0]
        if size != 0:
            avg_ssim = total_ssim / size
            avg_psnr = total_psnr / size
            return avg_ssim, avg_psnr
        return 0, 0

    def inverse(self, target_feature):
        """
        inverse the target feature.

        Args:
            target_feature (Tensor): The target feature.
        Returns:
            - Tensor, the reconstructed image.
        """
        target_feature = check_param_type('target_feature', target_feature, Tensor)
        return self._invnet(target_feature)
