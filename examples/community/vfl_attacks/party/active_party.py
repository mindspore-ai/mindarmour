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
import mindspore.nn as nn
import mindspore
import mindspore.ops as ops


class VFLActiveModel(object):
    """
    VFL active party
    """
    def __init__(self, bottom_model, args, top_model=None):
        super(VFLActiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False

        self.classifier_criterion = nn.CrossEntropyLoss()                   ###
        self.parties_grad_component_list = []  # latent representations from passive parties
        self.X = None
        self.y = None
        self.bottom_y = None  # latent representation from local bottom model
        self.top_grads = None  # gradients of local bottom model
        self.parties_grad_list = []  # gradients for passive parties
        self.epoch = None  # current train epoch
        self.indices = None  # indices of current train samples

        self.top_model = top_model
        self.top_trainable = True if self.top_model is not None else False

        self.args = args.copy()

        if self.args['cuda']:
            mindspore.set_context(device_target="GPU")          ####

        self.attack_indices = []
        self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
        self.bottom_gradient_function = self.grad_op(self.bottom_model, self.bottom_model.model.trainable_params())
        if self.top_trainable:
            self.top_grad_op = ops.value_and_grad(self.forward_fn, grad_position=0, weights=self.top_model.model.trainable_params())
        else:
            self.top_grad_op = ops.value_and_grad(self.forward_fn, grad_position=0, weights=None)

    def set_indices(self, indices):
        self.indices = indices

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch(self, X, y):
        self.X = X
        self.y = y

    def _fit(self, X, y):
        """
        compute gradients, and update local bottom model and top model

        :param X: features of active party
        :param y: labels
        """
        # get local latent representation
        self.bottom_y = self.bottom_model.forward(X)            #### forward
        self.K_U = self.bottom_y            #######
        
        # compute gradients based on labels, including gradients for passive parties
        self._compute_common_gradient_and_loss(y)

        # update parameters of local bottom model and top model
        self._update_models(X, y)

    def predict(self, X, component_list, type):
        """
        get the final prediction

        :param X: feature of active party
        :param component_list: latent representations from passive parties
        :return: prediction label
        """

        # get local latent representation
        U = self.bottom_model.forward(X)            #### forward


        # sum up latent representation in VFL without model splitting
        if not self.top_trainable:
            for comp in component_list:
                U = U + comp
        # use top model to predict in VFL with model splitting
        else:
            if self.args['aggregate'] == 'Concate':
                temp = ops.cat([U] + component_list, -1)              ###
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

        result = ops.softmax(U, axis=1)                    ###

        return result

    def receive_components(self, component_list):
        """
        receive latent representations from passive parties

        :param component_list: latent representations from passive parties
        """
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component)              #####

    def fit(self):
        """
        backward
        """
        self.parties_grad_list = []
        self._fit(self.X, self.y)
        self.parties_grad_component_list = []
        
    def forward_fn(self, top_input, y):     ### 新加
        """
        provide forward computation function for value_and_grad()
        
        :param y: label
        """
        if not self.top_trainable:
            U = top_input
        else:
            U = self.top_model.forward(top_input)

        class_loss = self.classifier_criterion(U, y)

        return class_loss

    def _compute_common_gradient_and_loss(self, y):
        """
        compute loss and gradients, including gradients for passive parties

        :param y: label
        """
        # compute prediction
        U = self.K_U

        grad_comp_list = [self.K_U] + self.parties_grad_component_list
        if not self.top_trainable:
            temp = U
            for grad_comp in self.parties_grad_component_list:
                temp = temp + grad_comp
        else:
            if self.args['aggregate'] == 'Concate':
                temp = ops.cat(grad_comp_list, -1)                    ###
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
        # compute gradients
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
            
        # save gradients of local bottom model
        self.top_grads = grad_list[0]
        # save gradients for passive parties
        for index in range(0, len(self.parties_grad_component_list)):
            parties_grad = grad_list[index+1]
            self.parties_grad_list.append(parties_grad)

        self.loss = class_loss.item()*self.K_U.shape[0]

    def send_gradients(self):
        """
        send gradients to passive parties
        """
        return self.parties_grad_list

    def _update_models(self, X, y):
        """
        update parameters of local bottom model and top model

        :param X: features of active party
        :param y: invalid
        """
        # update parameters of top model
        if self.top_trainable:
            self.top_model.backward_(self.para_grad_list)
        # self.bottom_model.backward(X, self.bottom_y, self.top_grads)
        self.bottom_backward(X, self.bottom_y, self.top_grads)

    def bottom_backward(self, x, y, grad_wrt_output):
        bottom_parma_grad = self.bottom_gradient_function(x, grad_wrt_output)
        self.bottom_model.backward_(bottom_parma_grad)


    def get_loss(self):
        return self.loss

    def save(self):
        """
        save model to local file
        """
        if self.top_trainable:
            self.top_model.save(time=self.args['file_time'])
        self.bottom_model.save(time=self.args['file_time'])

    def load(self):
        """
        load model from local file
        """
        if self.top_trainable:
            self.top_model.load(time=self.args['load_time'])
        self.bottom_model.load(time=self.args['load_time'])

    def set_train(self):
        """
        set train mode
        
        """
        if self.top_trainable:
            self.top_model.set_train(True)                 ###
        self.bottom_model.set_train(True)               ###

    def set_eval(self):
        """
        set eval mode
        """
        if self.top_trainable:
            self.top_model.set_train(False)              ###
        self.bottom_model.set_train(False)               ###

    def scheduler_step(self):
        """
        adjust learning rate during training
        """
        if self.top_trainable and self.top_model.scheduler is not None:
            self.top_model.scheduler.step()             ###
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()              ####

    def set_args(self, args):
        self.args = args

    def zero_grad(self):
        """
        clear gradients
        """
        pass

