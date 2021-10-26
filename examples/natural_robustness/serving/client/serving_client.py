# Copyright 2021 Huawei Technologies Co., Ltd
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
"""The client of example add."""
import os
import json
from io import BytesIO

import cv2
from PIL import Image
from mindspore_serving.client import Client

from perturb_config import PerturbConfig


def perturb(perturb_config):
    """Invoke servable perturbation method natural_perturbation"""
    client = Client("10.175.244.87:5500", "perturbation", "natural_perturbation")
    instances = []
    img_path = '/root/mindarmour/example/adversarial/test_data/1.png'
    result_path = '/root/mindarmour/example/adv/result/'
    methods_number = 2
    outputs_number = 3
    img = cv2.imread(img_path)
    img = cv2.imencode('.png', img)[1].tobytes()
    perturb_config = json.dumps(perturb_config)
    instances.append({"img": img, 'perturb_config': perturb_config, "methods_number": methods_number,
                      "outputs_number": outputs_number})

    result = client.infer(instances)

    file_names = result[0]['file_names'].split(';')
    length = result[0]['file_length'].tolist()
    before = 0
    for name, leng in zip(file_names, length):
        res_img = result[0]['results']
        res_img = res_img[before:before + leng]
        before = before + leng
        print('name: ', name)
        image = Image.open(BytesIO(res_img))
        image.save(os.path.join(result_path, name))


if __name__ == '__main__':
    perturb(PerturbConfig)
