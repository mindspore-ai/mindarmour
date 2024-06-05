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
Defines various functions of the active party in Vertical Federated Learning (VFL).

This module includes the implementation of various operations performed by the active party,
such as receiving components, training models, sending gradients, and other related tasks.
"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class VFLActiveModel(object):
    """
    VFL active party.
    """
    def __init__(self, bottom_model, args, top_model=None):
        super(VFLActiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.parties_grad_component_list = []
        self.X = None
        self.y = None
        self.bottom_y = None
        self.top_grads = None
        self.parties_grad_list = []
        self.epoch = None
        self.indices = None

        self.top_model = top_model
        self.top_trainable = True if self.top_model is not None else False

        self.args = args.copy()

        if self.args['cuda']:
            mindspore.set_context(device_target="GPU")

        self.attack_indices = []
        self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
        self.bottom_gradient_function = self.grad_op(self.bottom_model, self.bottom_model.model.trainable_params())
        if self.top_trainable:
            self.top_grad_op = ops.value_and_grad(self.forward_fn, grad_position=0,
                                                  weights=self.top_model.model.trainable_params())
        else:
            self.top_grad_op = ops.value_and_grad(self.forward_fn, grad_position=0, weights=None)

    def set_indices(self, indices):
        """
        Set the current indices for the active party.

        Args:
            indices (List[int]): A list of sample indices to be set as the current indices
                                 for the active party.
        """
        self.indices = indices

    def set_epoch(self, epoch):
        """
        Set the current epoch for the active party.
        Args:
            epoch (int): The current epoch for the active party.
        """
        self.epoch = epoch

    def set_batch(self, X, y):
        """
        Set the data and labels for the active party.

        Args:
            X (Tensor): The input data for the active party.
            y (Tensor): The labels corresponding to the input data.
        """
        self.X = X
        self.y = y

    def _fit(self, X, y):
        """
        Compute gradients and update the local bottom model and top model.

        Args:
            X (Tensor):The input data of the active party.
            y (Tensor): Labels corresponding to the input data.
        """
        # Get local latent representation
        self.bottom_y = self.bottom_model.forward(X)
        self.K_U = self.bottom_y
        
        # Compute gradients based on labels, including gradients for passive parties
        self._compute_common_gradient_and_loss(y)

        # Update parameters of local bottom model and top model
        self._update_models(X, y)

    def predict(self, X, component_list, type):
        """
        Get the final prediction.

        Args:
            X (Tensor): Feature of the active party.
            component_list (List[Tensor]): Latent representations from passive parties.

        Returns:
            Tensor: Predicted labels.
        """
        # Get local latent representation
        U = self.bottom_model.forward(X)


        # Sum up latent representation in VFL without model splitting
        if not self.top_trainable:
            for comp in component_list:
                U = U + comp
        # Use top model to predict in VFL with model splitting
        else:
            if self.args['aggregate'] == 'Concate':
                temp = ops.cat([U] + component_list, -1)
            elif self.args['aggregate'] == 'Add':
                temp = U
                for comp in component_list:
                    temp = temp + comp
            elif self.args['aggregate'] == 'Mean':
                temp = U
                for comp in component_list:
                    temp = temp + comp
                temp = temp / (len(component_list)+1)
            U = self.top_model.forward(temp)
        result = ops.softmax(U, axis=1)
        return result

    def receive_components(self, component_list):
        """
        Receive latent representations from passive parties.

        Args:
            component_list (List[Tensor]): Latent representations from passive parties.
        """
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component)

    def fit(self):
        """
        Backward.
        """
        self.parties_grad_list = []
        self._fit(self.X, self.y)
        self.parties_grad_component_list = []
        
    def forward_fn(self, top_input, y):
        """
        Provide a forward computation function for the `value_and_grad()` function.

        Args:
            y (Tensor): The label tensor used for computing the forward pass.
        """
        if not self.top_trainable:
            U = top_input
        else:
            U = self.top_model.forward(top_input)

        class_loss = self.classifier_criterion(U, y)

        return class_loss

    def _compute_common_gradient_and_loss(self, y):
        """
        Compute loss and gradients, including gradients for passive parties.

        Args:
            y (Tensor): The label tensor used for computing the loss and gradients.
        """
        U = self.K_U

        grad_comp_list = [self.K_U] + self.parties_grad_component_list
        if not self.top_trainable:
            temp = U
            for grad_comp in self.parties_grad_component_list:
                temp = temp + grad_comp
        else:
            if self.args['aggregate'] == 'Concate':
                temp = ops.cat(grad_comp_list, -1)
            elif self.args['aggregate'] == 'Add':
                temp = grad_comp_list[0]
                for comp in grad_comp_list[1:]:
                    temp = temp + comp
            elif self.args['aggregate'] == 'Mean':
                temp = grad_comp_list[0]
                for comp in grad_comp_list[1:]:
                    temp = temp + comp
                temp = temp / len(grad_comp_list)

        top_input = temp
        # Compute gradients.
        if self.top_trainable:
            class_loss, (top_input_grad, para_grad_list) = self.top_grad_op(top_input, y)
            self.para_grad_list = para_grad_list
            grad_list = []
            if self.args['aggregate'] == 'Concate':
                length = self.bottom_model.output_dim
                for i in range(len(self.parties_grad_component_list) + 1):
                    grad_list.append(top_input_grad[:, i*length:(i+1)*length])
            else:
                if self.args['aggregate'] == 'Mean':
                    for i in range(len(self.parties_grad_component_list) + 1):
                        grad_list.append(top_input_grad / (len(self.parties_grad_component_list) + 1))
                else:
                    for i in range(len(self.parties_grad_component_list) + 1):
                        grad_list.append(top_input_grad)
        else:
            class_loss, top_input_grad = self.top_grad_op(top_input,y)
            grad_list = []
            for i in range(len(self.parties_grad_component_list) + 1):
                grad_list.append(top_input_grad)
            
        # Save gradients of local bottom model.
        self.top_grads = grad_list[0]
        # Save gradients for passive parties.
        for index in range(0, len(self.parties_grad_component_list)):
            parties_grad = grad_list[index+1]
            self.parties_grad_list.append(parties_grad)

        self.loss = class_loss.item()*self.K_U.shape[0]

    def send_gradients(self):
        """
        Send gradients to passive parties.

        Returns:
            List[Tensor]: A list of gradient tensors to be sent to passive parties.
        """
        return self.parties_grad_list

    def _update_models(self, X, y):
        """
        Update parameters of the local bottom model and top model.

        Args:
            X (Tensor): Features of the active party.
            y (Tensor): The labels.
        """
        if self.top_trainable:
            self.top_model.backward_(self.para_grad_list)
        self.bottom_backward(X, self.bottom_y, self.top_grads)

    def bottom_backward(self, x, y, grad_wrt_output):
        """
        Update the bottom model.

        Args:
            x (Tensor): The input data.
            y (Tensor): The model output.
            grad_wrt_output (Tensor): The gradients with respect to the output.
        """
        bottom_parma_grad = self.bottom_gradient_function(x, grad_wrt_output)
        self.bottom_model.backward_(bottom_parma_grad)

    def get_loss(self):
        return self.loss

    def save(self):
        """
        Save model to local file.
        """
        if self.top_trainable:
            self.top_model.save(time=self.args['file_time'])
        self.bottom_model.save(time=self.args['file_time'])

    def load(self):
        """
        Load model from local file.
        """
        if self.top_trainable:
            self.top_model.load(time=self.args['load_time'])
        self.bottom_model.load(time=self.args['load_time'])

    def set_train(self):
        """
        Set train mode.
        """
        if self.top_trainable:
            self.top_model.set_train(True)
        self.bottom_model.set_train(True)

    def set_eval(self):
        """
        Set eval mode.
        """
        if self.top_trainable:
            self.top_model.set_train(False)
        self.bottom_model.set_train(False)

    def scheduler_step(self):
        """
        Adjust learning rate during training.
        """
        if self.top_trainable and self.top_model.scheduler is not None:
            self.top_model.scheduler.step()
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()

    def set_args(self, args):
        """
        Set arguments for the active party.

        Args:
            args (dict): A dictionary of arguments for the active party.
        """
        self.args = args

    def zero_grad(self):
        """
        Clear gradients.
        """
        pass
