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
Defines various functions of the passive party in Vertical Federated Learning (VFL).

This module includes the implementation of various functions performed by the passive party,
such as calculating and uploading intermediate data, receiving gradients, updating models, and other related tasks.
"""
from mindspore import ops

class VFLPassiveModel(object):
    """
    VFL passive party.
    """
    def __init__(self, bottom_model, id=None, args=None):
        super(VFLPassiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False
        self.common_grad = None
        self.X = None
        self.indices = None
        self.epoch = None
        self.y = None
        self.id = id
        self.args = args
        self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
        self.bottom_gradient_function = self.grad_op(self.bottom_model, self.bottom_model.model.trainable_params())

    def set_epoch(self, epoch):
        """
        Set the current epoch for the passive party.
        Args:
            epoch (int): The current epoch for the passive party.
        """
        self.epoch = epoch

    def set_batch(self, X, indices):
        """
        Set the current data and indices for the passive party.

        Args:
            X (Tensor): The input data for the passive party.
            indices (List[int]): A list of sample indices to be set as the current indices
                                 for the passive party.
        """
        self.X = X
        self.indices = indices

    def _forward_computation(self, X, model=None):
        """
        Perform the forward computation.

        Args:
            X (Tensor): Features of the passive party.
            model (Model): The model object, which is marked as invalid in this context.

        Returns:
            Tensor: The latent representation of the passive party.
        """
        if model is None:
            A_U = self.bottom_model.forward(X)
        else:
            A_U = model.forward(X)
        self.y = A_U
        return A_U

    def _fit(self, X, y):
        """
        Backward.

        Args:
            X (Tensor): Features of the passive party.
            y (Tensor): The latent representation of the passive party.
        """
        self.bottom_backward(X, y, self.common_grad, self.epoch)
        return

    def bottom_backward(self, x, y, grad_wrt_output, epoch):
        """
        Update the bottom model.

        Args:
            x (Tensor): The input data.
            y (Tensor): The model output.
            grad_wrt_output (Tensor): The gradients with respect to the output.
            epoch (int): The current epoch.
        """
        bottom_parma_grad = self.bottom_gradient_function(x, grad_wrt_output)
        self.bottom_model.backward_(bottom_parma_grad)

    def receive_gradients(self, gradients):
        """
        Receive gradients from the active party and update parameters of the local bottom model.

        Args:
            gradients (List[Tensor]): Gradients from the active party.
        """
        self.common_grad = gradients
        self._fit(self.X, self.y)

    def send_components(self):
        """
        Send latent representation to the active party.
        """
        result = self._forward_computation(self.X)
        return result

    def predict(self, X, is_attack=False):
        """
        Compute the output.

        Args:
            X (Tensor): Input data.
            is_attack (bool): Indicates whether the computation is for an attack scenario or not.

        Returns:
            Tensor: Embeddings to be sent to the active party.
        """
        return self._forward_computation(X)

    def save(self):
        """
        Save model to local file.
        """
        self.bottom_model.save(id=self.id, time=self.args['file_time'])

    def load(self, load_attack=False):
        """
        Load the model from a local file.

        Args:
            load_attack (bool): A flag indicating whether to load the attack model, marked as invalid in this context.
        """
        if load_attack:
            self.bottom_model.load(name='attack', time=self.args['load_time'])
        else:
            self.bottom_model.load(id=self.id, time=self.args['load_time'])

    def set_train(self):
        """
        Set train mode.
        """
        self.bottom_model.set_train(True)

    def set_eval(self):
        """
        Set eval mode.
        """
        self.bottom_model.set_train(False)

    def scheduler_step(self):
        """
        Adjust learning rate during training.
        """
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()

    def zero_grad(self):
        """
        Clear gradients.
        """
        pass
