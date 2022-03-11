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
"""cnnctc eval"""

import numpy as np
import lmdb
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.dataset import GeneratorDataset
from cnn_ctc.src.util import CTCLabelConverter
from cnn_ctc.src.dataset import iiit_generator_batch, adv_iiit_generator_batch
from cnn_ctc.src.cnn_ctc import CNNCTC
from cnn_ctc.src.model_utils.config import config

context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                    save_graphs_path=".")


def test_dataset_creator(is_adv=False):
    if is_adv:
        ds = GeneratorDataset(adv_iiit_generator_batch(), ['img', 'label_indices', 'text',
                                                           'sequence_length', 'label_str'])
    else:
        ds = GeneratorDataset(iiit_generator_batch, ['img', 'label_indices', 'text',
                                                     'sequence_length', 'label_str'])
    return ds


def test(lmdb_save_path):
    """eval cnnctc model on begin and perturb data."""
    target = config.device_target
    context.set_context(device_target=target)

    ds = test_dataset_creator(is_adv=config.IS_ADV)
    net = CNNCTC(config.NUM_CLASS, config.HIDDEN_SIZE, config.FINAL_FEATURE_WIDTH)

    ckpt_path = config.CHECKPOINT_PATH
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    print('parameters loaded! from: ', ckpt_path)

    converter = CTCLabelConverter(config.CHARACTER)

    count = 0
    correct_count = 0
    env_save = lmdb.open(lmdb_save_path, map_size=1099511627776)
    with env_save.begin(write=True) as txn_save:
        for data in ds.create_tuple_iterator():
            img, _, text, _, length = data

            img_tensor = Tensor(img, mstype.float32)

            model_predict = net(img_tensor)
            model_predict = np.squeeze(model_predict.asnumpy())

            preds_size = np.array([model_predict.shape[1]] * config.TEST_BATCH_SIZE)
            preds_index = np.argmax(model_predict, 2)
            preds_index = np.reshape(preds_index, [-1])
            preds_str = converter.decode(preds_index, preds_size)
            label_str = converter.reverse_encode(text.asnumpy(), length.asnumpy())

            print("Prediction samples: \n", preds_str[:5])
            print("Ground truth: \n", label_str[:5])
            for pred, label in zip(preds_str, label_str):
                if pred == label:
                    correct_count += 1
                count += 1
                if config.IS_ADV:
                    pred_key = 'adv_pred-%09d'.encode() % count
                else:
                    pred_key = 'pred-%09d'.encode() % count

                txn_save.put(pred_key, pred.encode())
    accuracy = correct_count / count
    return accuracy


if __name__ == '__main__':
    save_path = config.ADV_TEST_DATASET_PATH
    config.IS_ADV = False
    config.TEST_DATASET_PATH = save_path
    ori_acc = test(lmdb_save_path=save_path)

    config.IS_ADV = True
    adv_acc = test(lmdb_save_path=save_path)
    print('Accuracy of benign sample: ', ori_acc)
    print('Accuracy of perturbed sample: ', adv_acc)
