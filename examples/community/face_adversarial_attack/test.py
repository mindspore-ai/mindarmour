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
"""test"""
import numpy as np
from mindspore import context, Tensor
import mindspore
from mindspore.dataset.vision.py_transforms import ToTensor
import mindspore.dataset.vision.py_transforms as P
from FaceRecognition.eval import get_model
import adversarial_attack

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

if __name__ == '__main__':

    image = adversarial_attack.load_data('photos/adv_input/')
    inputs = adversarial_attack.load_data('photos/input/')
    targets = adversarial_attack.load_data('photos/target/')

    tensorize = ToTensor()
    normalize = P.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    expand_dims = mindspore.ops.ExpandDims()
    mean = Tensor([0.485, 0.456, 0.406])
    std = Tensor([0.229, 0.224, 0.225])

    resnet = get_model()

    adv = Tensor(normalize(tensorize(image[0])))
    input_tensor = Tensor(normalize(tensorize(inputs[0])))
    target_tensor = Tensor(normalize(tensorize(targets[0])))

    adversarial_emb = resnet(expand_dims(adv, 0))
    input_emb = resnet(expand_dims(input_tensor, 0))
    target_emb = resnet(expand_dims(target_tensor, 0))

    adversarial_index = np.argmax(adversarial_emb.asnumpy())
    target_index = np.argmax(target_emb.asnumpy())
    input_index = np.argmax(input_emb.asnumpy())

    print("input_label:", input_index)
    print("The confidence of the input image on the input label:", input_emb.asnumpy()[0][input_index])
    print("================================")
    print("adversarial_label:", adversarial_index)
    print("The confidence of the adversarial sample on the correct label:", adversarial_emb.asnumpy()[0][input_index])
    print("The confidence of the adversarial sample on the adversarial label:",
          adversarial_emb.asnumpy()[0][adversarial_index])
    print("input_label:%d, adversarial_label:%d" % (input_index, adversarial_index))
