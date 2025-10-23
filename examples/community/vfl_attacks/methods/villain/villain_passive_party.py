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
Malicious passive party for Villain backdoor
"""
import random
import mindspore as ms
from mindspore import ops, Parameter
from MindsporeCode.party.passive_party import VFLPassiveModel
from MindsporeCode.common.constants import CHECKPOINT_PATH
import os



class Villain_PassiveModel(VFLPassiveModel):
    def __init__(self, bottom_model, amplify_ratio=1, args=None, id=None):
        super(Villain_PassiveModel, self).__init__(bottom_model, id=id, args=args)
        self.target_grad = None
        self.target_indices = None
        self.amplify_ratio = amplify_ratio
        self.components = None
        self.is_debug = False
        self.pair_set = dict()
        self.target_gradients = dict()
        self.backdoor_X = dict()
        self.args = args

        self.attack = False
        self.feature_pattern = None
        self.mask = None
        self.drop_out = True
        self.drop_out_rate = 0.75
        self.max_norms = None
        self.shifting = True
        self.up_bound = 1.2
        self.down_bound = 0.6
        self.adversary = self.args['adversary'] - 1

    def save_data(self, name=None, pattern=None):
        """
        save the pixel pattern or feature pattern
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.args['dataset'], self.args['file_time'])
        if not os.path.exists(path):
            os.makedirs(path)
        name = self.args['trigger'] + '_pattern.ckpt'  # 昇思通常使用.ckpt作为检查点文件的扩展名
        filepath = '{}/{}'.format(path, name)
        if self.args['trigger'] == 'feature':
            ms.save_checkpoint(self.feature_pattern, filepath)
        else:
            ms.save_checkpoint(pattern, filepath)

    def load_data(self, name=None):
        """
        save the pixel pattern or feature pattern
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.args['dataset'], self.args['load_time'])
        name = self.args['trigger'] + '_pattern.ckpt'
        filepath = '{}/{}'.format(path, name)
        if os.path.isfile(filepath):
            pattern = ms.load_checkpoint(filepath)
            if self.args['trigger'] == 'feature':
                self.feature_pattern = pattern
            return pattern
        else:
            raise ValueError("load data error, wrong filepath")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_backdoor_indices(self, target_indices):
        self.target_indices = target_indices

    def set_batch(self, X, indices):
        self.X = X
        self.indices = indices

    def receive_gradients(self, gradients):
        gradients = gradients.copy()
        gradients = self.amplify_ratio * gradients
        self.common_grad = gradients
        # backwards
        self._fit(self.X, self.components)
        return

    def send_components(self):
        result = self._forward_computation(self.X)
        self.components = result
        send_result = result.copy()

        for index, i in enumerate(self.indices):
            if self.target_indices is not None and i.item() in self.target_indices:
                if self.feature_pattern is not None:
                    self.mask = ops.full_like(send_result[index], 1)
                    if self.drop_out:
                        self.mask = ops.dropout(self.mask, p=self.drop_out_rate, training=True)
                        self.mask = self.mask * (1 - self.drop_out_rate)
                    if self.shifting:
                        num = random.random() * (self.up_bound - self.down_bound) + self.down_bound
                        send_result[index] += self.feature_pattern * self.mask * num
                    else:
                        send_result[index] += self.feature_pattern * self.mask
        return send_result

    def predict(self, X, is_attack=False):
        result = self._forward_computation(X)
        send_results = result
        if is_attack and self.feature_pattern is not None:
            for i in range(len(send_results)):
                send_results[i] += self.feature_pattern

        return send_results
