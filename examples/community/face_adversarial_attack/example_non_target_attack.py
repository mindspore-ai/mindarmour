# Copyright 2022 Huawei Technologies Co., Ltd
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
"""non target attack"""
import numpy as np
import matplotlib.image as mp
from mindspore import context
import adversarial_attack
from FaceRecognition.eval import get_model

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


if __name__ == '__main__':

    inputs = adversarial_attack.load_data('photos/input/')
    targets = adversarial_attack.load_data('photos/target/')

    net = get_model()
    adversarial = adversarial_attack.FaceAdversarialAttack(inputs[0], targets[0], net)
    ATTACK_METHOD = "non_target_attack"

    tensor_dict = adversarial.train(attack_method=ATTACK_METHOD)

    mp.imsave('./outputs/adversarial_example.jpg',
              np.transpose(tensor_dict.get("adversarial_tensor").asnumpy(), (1, 2, 0)))
    mp.imsave('./outputs/mask.jpg',
              np.transpose(tensor_dict.get("mask_tensor").asnumpy(), (1, 2, 0)))
    mp.imsave('./outputs/input_image.jpg',
              np.transpose(tensor_dict.get("processed_input_tensor").asnumpy(), (1, 2, 0)))
    mp.imsave('./outputs/target_image.jpg',
              np.transpose(tensor_dict.get("processed_target_tensor").asnumpy(), (1, 2, 0)))

    adversarial.test_non_target_attack()
