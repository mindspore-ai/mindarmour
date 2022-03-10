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

"""Analyse result of ocr evaluation."""

import os
import sys
import json
from collections import defaultdict
from io import BytesIO
import lmdb
from PIL import Image

from cnn_ctc.src.model_utils.config import config


def analyse_adv_iii5t_3000(lmdb_path):
    """Analyse result of ocr evaluation."""
    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    if not env:
        print('cannot create lmdb from %s' % (lmdb_path))
        sys.exit(0)

    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
        print(n_samples)
        n_samples = n_samples // config.TEST_BATCH_SIZE * config.TEST_BATCH_SIZE
        result = defaultdict(dict)
        wrong_count = 0
        adv_wrong_count = 0
        ori_correct_adv_wrong_count = 0
        ori_wrong_adv_wrong_count = 0
        if not os.path.exists(os.path.join(lmdb_path, 'adv_wrong_pred')):
            os.mkdir(os.path.join(lmdb_path, 'adv_wrong_pred'))
        if not os.path.exists(os.path.join(lmdb_path, 'ori_correct_adv_wrong_pred')):
            os.mkdir(os.path.join(lmdb_path, 'ori_correct_adv_wrong_pred'))
        if not os.path.exists(os.path.join(lmdb_path, 'ori_wrong_adv_wrong_pred')):
            os.mkdir(os.path.join(lmdb_path, 'ori_wrong_adv_wrong_pred'))

        for index in range(n_samples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8').lower()
            pred_key = 'pred-%09d'.encode() % index
            pred = txn.get(pred_key).decode('utf-8')
            if pred != label:
                wrong_count += 1

            adv_pred_key = 'adv_pred-%09d'.encode() % index
            adv_pred = txn.get(adv_pred_key).decode('utf-8')

            adv_info_key = 'adv_info-%09d'.encode() % index
            adv_info = json.loads(txn.get(adv_info_key).decode('utf-8'))
            for info in adv_info:
                if not result[info[0]]:
                    result[info[0]] = defaultdict(int)
                result[info[0]]['count'] += 1

            if adv_pred != label:
                adv_wrong_count += 1
                for info in adv_info:
                    result[info[0]]['wrong_count'] += 1

                # save wrong predicted image
                adv_image = 'adv_image-%09d'.encode() % index
                imgbuf = txn.get(adv_image)
                image = Image.open(BytesIO(imgbuf))

                result_path = os.path.join(lmdb_path, 'adv_wrong_pred', adv_info[0][0])
                if not os.path.exists(result_path):
                    os.mkdir(result_path)

                image.save(os.path.join(result_path, label + '-' + adv_pred + '.png'))

                # origin image is correctly predicted and adv is wrong.
                if pred == label:
                    ori_correct_adv_wrong_count += 1
                    result[info[0]]['ori_correct_adv_wrong_count'] += 1

                    result_path = os.path.join(lmdb_path, 'ori_correct_adv_wrong_pred', adv_info[0][0])
                    if not os.path.exists(result_path):
                        os.mkdir(result_path)
                    image.save(os.path.join(result_path, label + '-' + adv_pred + '.png'))
                # wrong predicted in both origin and adv image.
                else:
                    ori_wrong_adv_wrong_count += 1
                    result[info[0]]['ori_wrong_adv_wrong_count'] += 1

                    result_path = os.path.join(lmdb_path, 'ori_wrong_adv_wrong_pred', adv_info[0][0])
                    if not os.path.exists(result_path):
                        os.mkdir(result_path)
                    image.save(os.path.join(result_path, label + '-' + adv_pred + '.png'))
    print('Number of samples in analyse dataset: ', n_samples)
    print('Accuracy of original dataset: ', 1 - wrong_count / n_samples)
    print('Accuracy of adversarial dataset: ', 1 - adv_wrong_count / n_samples)
    print('Number of samples correctly predicted in original dataset but wrong in adversarial dataset: ',
          ori_correct_adv_wrong_count)
    print('Number of samples both wrong predicted in original and adversarial dataset: ', ori_wrong_adv_wrong_count)
    print('------------------------------------------------------------------------------')
    for key in result.keys():
        print('Method ', key)
        print('Number of perturb samples: {} '.format(result[key]['count']))
        print('Number of wrong predicted: {}'.format(result[key]['wrong_count']))
        print('Number of correctly predicted in origin dataset but wrong in adversarial: {}'.format(
            result[key]['ori_correct_adv_wrong_count']))
        print('Number of both wrong predicted in origin and adversarial dataset: {}'.format(
            result[key]['ori_wrong_adv_wrong_count']))
        print('------------------------------------------------------------------------------')
    return result


if __name__ == '__main__':
    lmdb_data_path = config.ADV_TEST_DATASET_PATH
    analyse_adv_iii5t_3000(lmdb_path=lmdb_data_path)
