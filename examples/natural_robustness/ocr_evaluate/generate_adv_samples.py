# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Generated natural robustness samples. """

import sys
import json
import time
import lmdb
from mindspore_serving.client import Client
from cnn_ctc.src.model_utils.config import config

config_perturb = [
    {"method": "Contrast", "params": {"alpha": 1.5, "beta": 0}},
    {"method": "GaussianBlur", "params": {"ksize": 5}},
    {"method": "SaltAndPepperNoise", "params": {"factor": 0.05}},
    {"method": "Translate", "params": {"x_bias": 0.1, "y_bias": -0.1}},
    {"method": "Scale", "params": {"factor_x": 0.8, "factor_y": 0.8}},
    {"method": "Shear", "params": {"factor": 1.5, "direction": "horizontal"}},
    {"method": "Rotate", "params": {"angle": 30}},
    {"method": "MotionBlur", "params": {"degree": 5, "angle": 45}},
    {"method": "GradientBlur", "params": {"point": [50, 100], "kernel_num": 3, "center": True}},
    {"method": "GradientLuminance", "params": {"color_start": [255, 255, 255], "color_end": [0, 0, 0],
                                               "start_point": [100, 150], "scope": 0.3,
                                               "bright_rate": 0.3, "pattern": "light", "mode": "circle"}},
    {"method": "GradientLuminance", "params": {"color_start": [255, 255, 255],
                                               "color_end": [0, 0, 0], "start_point": [150, 200],
                                               "scope": 0.3, "pattern": "light", "mode": "horizontal"}},
    {"method": "GradientLuminance", "params": {"color_start": [255, 255, 255], "color_end": [0, 0, 0],
                                               "start_point": [150, 200], "scope": 0.3,
                                               "pattern": "light", "mode": "vertical"}},
    {"method": "Curve", "params": {"curves": 0.5, "depth": 3, "mode": "vertical"}},
    {"method": "Perspective", "params": {"ori_pos": [[0, 0], [0, 800], [800, 0], [800, 800]],
                                         "dst_pos": [[10, 0], [0, 800], [790, 0], [800, 800]]}},
]


def generate_adv_iii5t_3000(lmdb_paths, lmdb_save_path, perturb_config):
    """generate perturb iii5t_3000"""
    max_len = int((26 + 1) // 2)

    instances = []
    methods_number = 1
    outputs_number = 2
    perturb_config = json.dumps(perturb_config)

    env = lmdb.open(lmdb_paths, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    if not env:
        print('cannot create lmdb from %s' % (lmdb_paths))
        sys.exit(0)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))

        # Filtering
        filtered_labels = []
        filtered_index_list = []
        for index in range(n_samples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > max_len: continue
            illegal_sample = False
            for char_item in label.lower():
                if char_item not in config.CHARACTER:
                    illegal_sample = True
                    break
            if illegal_sample: continue

            filtered_labels.append(label)
            filtered_index_list.append(index)
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            instances.append({"img": imgbuf, 'perturb_config': perturb_config, "methods_number": methods_number,
                              "outputs_number": outputs_number})

    print(f'num of samples in IIIT dataset: {len(filtered_index_list)}')

    client = Client("0.0.0.0:5500", "perturbation", "natural_perturbation")
    start_time = time.time()
    result = client.infer(instances)
    end_time = time.time()
    print('generated natural perturbs images cost: ', end_time - start_time)
    env_save = lmdb.open(lmdb_save_path, map_size=1099511627776)

    txn = env.begin(write=False)
    with env_save.begin(write=True) as txn_save:
        new_index = 1
        for i, index in enumerate(filtered_index_list):
            try:
                file_names = result[i]['file_names'].split(';')
            except KeyError:
                error_msg = result[i]
                msg = 'serving failed to generate the {}th image in origin dataset with ' \
                      'error messages: {}'.format(i, error_msg)
                print(KeyError(msg))
                continue

            length = result[i]['file_length'].tolist()
            before = 0
            label = filtered_labels[i]
            label = label.encode()
            img_key = 'image-%09d'.encode() % index
            ori_img = txn.get(img_key)

            names_dict = result[i]['names_dict']
            names_dict = json.loads(names_dict)
            for name, leng in zip(file_names, length):
                label_key = 'label-%09d'.encode() % new_index
                txn_save.put(label_key, label)
                img_key = 'image-%09d'.encode() % new_index

                adv_img = result[i]['results']
                adv_img = adv_img[before:before + leng]
                adv_img_key = 'adv_image-%09d'.encode() % new_index
                txn_save.put(img_key, ori_img)
                txn_save.put(adv_img_key, adv_img)

                adv_info_key = 'adv_info-%09d'.encode() % new_index
                adv_info = json.dumps(names_dict[name]).encode()
                txn_save.put(adv_info_key, adv_info)
                before = before + leng
                new_index += 1
        txn_save.put("num-samples".encode(), str(new_index - 1).encode())
    env.close()


if __name__ == '__main__':
    save_path_lmdb = config.ADV_TEST_DATASET_PATH
    generate_adv_iii5t_3000(config.TEST_DATASET_PATH, save_path_lmdb, config_perturb)
