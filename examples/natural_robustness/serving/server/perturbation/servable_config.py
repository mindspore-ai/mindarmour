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
"""perturbation servable config"""
import json
import copy
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
from mindspore_serving.server import register
from mindarmour.natural_robustness.natural_noise import *

# Path of template images
TEMPLATE_LEAF_PATH = '/root/mindarmour/example/adv/test_data/template/leaf'
TEMPLATE_WINDOW_PATH = '/root/mindarmour/example/adv/test_data/template/window'
TEMPLATE_PERSON_PATH = '/root/mindarmour/example/adv/test_data/template/person'
TEMPLATE_BACKGROUND_PATH = '/root/mindarmour/example/adv/test_data//template/dirt_background'

path_dict = {'leaf': TEMPLATE_LEAF_PATH,
             'window': TEMPLATE_WINDOW_PATH,
             'person': TEMPLATE_PERSON_PATH,
             'background': TEMPLATE_BACKGROUND_PATH}

methods_dict = {'Contrast': Contrast,
                'GaussianBlur': GaussianBlur,
                'SaltAndPepperNoise': SaltAndPepperNoise,
                'Translate': Translate,
                'Scale': Scale,
                'Shear': Shear,
                'Rotate': Rotate,
                'MotionBlur': MotionBlur,
                'GradientBlur': GradientBlur,
                'GradientLuminance': GradientLuminance,
                'Perlin': Perlin,
                'BackShadow': BackShadow,
                'NaturalNoise': NaturalNoise,
                'Curve': Curve,
                'BackgroundWord': BackgroundWord,
                'Perspective': Perspective}


def check_inputs(img, perturb_config, methods_number, outputs_number):
    """Check inputs."""
    if not np.any(img):
        raise ValueError("img cannot be empty.")
    img = Image.open(BytesIO(img))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    config = json.loads(perturb_config)
    if not config:
        raise ValueError("perturb_config cannot be empty.")
    for item in config:
        if item['method'] not in methods_dict.keys():
            raise ValueError("{} is not a valid method.".format(item['method']))
        if item['method'] == 'BackShadow':
            item['params']['template_path'] = path_dict[item['params']['back_type']]
            del item['params']['back_type']

    methods_number = int(methods_number)
    if methods_number < 1:
        raise ValueError("methods_number must more than 0.")
    outputs_number = int(outputs_number)
    if outputs_number < 1:
        raise ValueError("outputs_number must more than 0.")

    return img, config, methods_number, outputs_number


def perturb(img, perturb_config, methods_number, outputs_number):
    img, config, methods_number, outputs_number = check_inputs(img, perturb_config, methods_number, outputs_number)
    res_img_bytes = b''
    file_names = []
    file_length = []
    for _ in range(outputs_number):
        file_name = ''
        dst = copy.deepcopy(img)
        for _ in range(methods_number):
            item = np.random.choice(config)
            method_name = item['method']
            method = methods_dict[method_name]
            params = item['params']
            dst = method(**params)(img)

            file_name = file_name + method_name + '_'
            for key in params:
                if key == 'template_path':
                    file_name += 'back_type_'
                    file_name += params[key].split('/')[-1]
                    file_name += '_'
                    continue
                file_name += key
                file_name += '_'
                file_name += str(params[key])
                file_name += '_'
            file_name += '#'

        file_name += '.png'
        file_names.append(file_name)

        res_img = cv2.imencode('.png', dst)[1].tobytes()
        res_img_bytes += res_img
        file_length.append(len(res_img))

    return res_img_bytes, ';'.join(file_names), file_length


model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)


@register.register_method(output_names=["results", "file_names", "file_length"])
def natural_perturbation(img, perturb_config, methods_number, outputs_number):
    """method natural_perturbation data flow definition, only preprocessing and call model"""
    res = register.add_stage(perturb, img, perturb_config, methods_number, outputs_number, outputs_count=3)
    return res
