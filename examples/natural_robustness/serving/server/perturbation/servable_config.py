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
import random
from io import BytesIO
import cv2
from PIL import Image
from mindspore_serving.server import register
from mindarmour.natural_robustness.natural_noise import *

# Path of template images
TEMPLATE_LEAF_PATH = '/root/mindarmour/example/adv/test_data/template/leaf'
TEMPLATE_WINDOW_PATH = '/root/mindarmour/example/adv/test_data/template/window'
TEMPLATE_PERSON_PATH = '/root/mindarmour/example/adv/test_data/template/person'
TEMPLATE_BACKGROUND_PATH = '/root/mindarmour/example/adv/test_data//template/dirt_background'

CHARACTERS = [chr(i) for i in range(65, 91)]+[chr(j) for j in range(97, 123)]

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
    """Perturb given image."""
    img, config, methods_number, outputs_number = check_inputs(img, perturb_config, methods_number, outputs_number)
    res_img_bytes = b''
    file_names = []
    file_length = []
    names_dict = {}
    for _ in range(outputs_number):
        dst = copy.deepcopy(img)
        used_methods = []
        for _ in range(methods_number):
            item = np.random.choice(config)
            method_name = item['method']
            method = methods_dict[method_name]
            params = item['params']
            dst = method(**params)(img)

            if method_name == 'BackShadow':
                method_params = copy.deepcopy(params)
                method_params['back_type'] = method_params['template_path'].split('/')[-1]
                del method_params['template_path']
            else:
                method_params = params

            used_methods.append([method_name, method_params])
        name = ''.join(random.sample(CHARACTERS, 20))
        name += '.png'
        file_names.append(name)
        names_dict[name] = used_methods

        res_img = cv2.imencode('.png', dst)[1].tobytes()
        res_img_bytes += res_img
        file_length.append(len(res_img))

    names_dict = json.dumps(names_dict)

    return res_img_bytes, ';'.join(file_names), file_length, names_dict


model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)


@register.register_method(output_names=["results", "file_names", "file_length", "names_dict"])
def natural_perturbation(img, perturb_config, methods_number, outputs_number):
    """method natural_perturbation data flow definition, only preprocessing and call model"""
    res = register.add_stage(perturb, img, perturb_config, methods_number, outputs_number, outputs_count=4)
    return res
