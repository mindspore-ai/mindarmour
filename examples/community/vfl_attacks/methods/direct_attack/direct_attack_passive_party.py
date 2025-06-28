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
Malicious passive party for direct label inference attack
"""
from MindsporeCode.party.passive_party import VFLPassiveModel
from MindsporeCode.common.image_report import append_int_to_file

class Direct_Attack_PassiveModel(VFLPassiveModel):
    def __init__(self, bottom_model, id=None, args=None):
        VFLPassiveModel.__init__(self, bottom_model, id, args)
        self.batch_label = None
        self.inferred_correct = 0
        self.inferred_wrong = 0

    def send_components(self):
        """
        send latent representation to active party
        """
        result = self._forward_computation(self.X)
        return result

    def receive_gradients(self, gradients):
        """
        receive gradients from active party and update parameters of local bottom model

        :param gradients: gradients from active party
        """
        # direct label inference attack
        for sample_id in range(len(gradients)):
            grad_per_sample = gradients[sample_id]
            for logit_id in range(len(grad_per_sample)):
                # true label
                if grad_per_sample[logit_id] < 0:
                    inferred_label = logit_id
                    if inferred_label == self.batch_label[sample_id]:
                        self.inferred_correct += 1
                    else:
                        self.inferred_wrong += 1
                    break
            # visualization image for web
            if self.epoch == 0 and sample_id == 0:
                output_txt_path = '../temp_output/' +  self.args['file_time'] + '-image-report.txt'
                with open(output_txt_path, "r") as file:
                    content = file.read()
                    if 'clean' not in content:
                        y_true, y_infer = self.batch_label[sample_id].item(), inferred_label
                        append_int_to_file(output_txt_path, y_true, y_infer)

        self.common_grad = gradients
        self._fit(self.X, self.y)

