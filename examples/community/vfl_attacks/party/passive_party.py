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

from mindspore import Tensor, ops, nn, Parameter

class VFLPassiveModel(object):
    """
    VFL passive party
    """
    def __init__(self, bottom_model, id=None, args=None):
        super(VFLPassiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False
        self.common_grad = None  # gradients
        self.X = None
        self.indices = None
        self.epoch = None
        self.y = None
        self.id = id  # id of passive party
        self.args = args
        self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
        self.bottom_gradient_function = self.grad_op(self.bottom_model, self.bottom_model.model.trainable_params())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch(self, X, indices):
        self.X = X
        self.indices = indices

    def _forward_computation(self, X, model=None):
        """
        forward

        :param X: features of passive party
        :param model: invalid
        :return: latent representation of passive party
        """
        if model is None:
            A_U = self.bottom_model.forward(X)
        else:
            A_U = model.forward(X)
        self.y = A_U
        return A_U

    def _fit(self, X, y):
        """
        backward

        :param X: features of passive party
        :param y: latent representation of passive party
        """
        self.bottom_backward(X, y, self.common_grad, self.epoch)
        return

    def bottom_backward(self, x, y, grad_wrt_output, epoch):
        bottom_parma_grad = self.bottom_gradient_function(x, grad_wrt_output)
        self.bottom_model.backward_(bottom_parma_grad)

    def receive_gradients(self, gradients):
        """
        receive gradients from active party and update parameters of local bottom model

        :param gradients: gradients from active party
        """
        self.common_grad = gradients
        self._fit(self.X, self.y)

    def send_components(self):
        """
        send latent representation to active party
        """
        result = self._forward_computation(self.X)
        return result

    def predict(self, X, is_attack=False):
        return self._forward_computation(X)

    def save(self):
        """
        save model to local file
        """
        self.bottom_model.save(id=self.id, time=self.args['file_time'])

    def load(self, load_attack=False):
        """
        load model from local file

        :param load_attack: invalid
        """
        if load_attack:
            self.bottom_model.load(name='attack', time=self.args['load_time'])
        else:
            self.bottom_model.load(id=self.id, time=self.args['load_time'])

    def set_train(self):
        """
        set train mode
        """
        self.bottom_model.set_train(True)                   #####

    def set_eval(self):
        """
        set eval mode
        """
        self.bottom_model.set_train(False)          #######

    def scheduler_step(self):
        """
        adjust learning rate during training
        """
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()              ######

    def zero_grad(self):
        """
        clear gradients
        """
        pass

