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
Define the passive party for direct label inference attack.

This module provides the implementation of the passive party in scenarios involving direct label inference attacks.
"""
from party.passive_party import VFLPassiveModel

class DirectAttackPassiveModel(VFLPassiveModel):
    """
    Malicious passive party for direct label inference attack.
    """
    def __init__(self, bottom_model, id=None, args=None):
        VFLPassiveModel.__init__(self, bottom_model, id, args)
        self.batch_label = None
        self.inferred_correct = 0
        self.inferred_wrong = 0

    def send_components(self):
        """
        Send latent representation to active party.
        """
        result = self._forward_computation(self.X)
        return result

    def receive_gradients(self, gradients):
        """
        Receive gradients from the active party and update parameters of the local bottom model.

        Args:
            gradients (Tensor): Gradients from the active party.
        """
        for sample_id in range(len(gradients)):
            grad_per_sample = gradients[sample_id]
            for logit_id in range(len(grad_per_sample)):
                if grad_per_sample[logit_id] < 0:
                    inferred_label = logit_id
                    if inferred_label == self.batch_label[sample_id]:
                        self.inferred_correct += 1
                    else:
                        self.inferred_wrong += 1
                    break
        self.common_grad = gradients
        self._fit(self.X, self.y)
