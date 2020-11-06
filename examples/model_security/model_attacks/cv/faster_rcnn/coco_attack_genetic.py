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
"""PSO attack for Faster R-CNN"""
import os
import numpy as np

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore import Tensor

from mindarmour import BlackModel
from mindarmour.adv_robustness.attacks.black.genetic_attack import GeneticAttack
from mindarmour.utils.logger import LogUtil

from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset

# pylint: disable=locally-disabled, unused-argument, redefined-outer-name
LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')

set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=1)


class ModelToBeAttacked(BlackModel):
    """model to be attack"""

    def __init__(self, network):
        super(ModelToBeAttacked, self).__init__()
        self._network = network

    def predict(self, images, img_metas, gt_boxes, gt_labels, gt_num):
        """predict"""
        # Adapt to the input shape requirements of the target network if inputs is only one image.
        if len(images.shape) == 3:
            inputs_num = 1
            images = np.expand_dims(images, axis=0)
        else:
            inputs_num = images.shape[0]
        box_and_confi = []
        pred_labels = []
        gt_number = np.sum(gt_num)

        for i in range(inputs_num):
            inputs_i = np.expand_dims(images[i], axis=0)
            output = self._network(Tensor(inputs_i.astype(np.float16)), Tensor(img_metas),
                                   Tensor(gt_boxes), Tensor(gt_labels), Tensor(gt_num))
            all_bbox = output[0]
            all_labels = output[1]
            all_mask = output[2]
            all_bbox_squee = np.squeeze(all_bbox.asnumpy())
            all_labels_squee = np.squeeze(all_labels.asnumpy())
            all_mask_squee = np.squeeze(all_mask.asnumpy())
            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_labels_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > gt_number + 1:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:gt_number+1]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]
            box_and_confi.append(all_bboxes_tmp_mask)
            pred_labels.append(all_labels_tmp_mask)
        return np.array(box_and_confi), np.array(pred_labels)


if __name__ == '__main__':
    prefix = 'FasterRcnn_eval.mindrecord'
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    pre_trained = '/ckpt_path'
    print("CHECKING MINDRECORD FILES ...")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if os.path.isdir(config.coco_root):
            print("Create Mindrecord. It may take some time.")
            data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("coco_root not exits.")

    print('Start generate adversarial samples.')

    # build network and dataset
    ds = create_fasterrcnn_dataset(mindrecord_file, batch_size=config.test_batch_size, \
                                    repeat_num=1, is_training=False)
    net = Faster_Rcnn_Resnet50(config)
    param_dict = load_checkpoint(pre_trained)
    load_param_into_net(net, param_dict)
    net = net.set_train(False)

    # build attacker
    model = ModelToBeAttacked(net)
    attack = GeneticAttack(model, model_type='detection', max_steps=50, reserve_ratio=0.3, mutation_rate=0.05,
                           per_bounds=0.5, step_size=0.25, temp=0.1)

    # generate adversarial samples
    sample_num = 5
    ori_imagess = []
    adv_imgs = []
    ori_meta = []
    ori_box = []
    ori_labels = []
    ori_gt_num = []
    idx = 0
    for data in ds.create_dict_iterator():
        if idx > sample_num:
            break
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        ori_imagess.append(img_data.asnumpy())
        ori_meta.append(img_metas.asnumpy())
        ori_box.append(gt_bboxes.asnumpy())
        ori_labels.append(gt_labels.asnumpy())
        ori_gt_num.append(gt_num.asnumpy())

        all_inputs = (img_data.asnumpy(), img_metas.asnumpy(), gt_bboxes.asnumpy(),
                      gt_labels.asnumpy(), gt_num.asnumpy())

        pre_gt_boxes, pre_gt_label = model.predict(*all_inputs)
        success_flags, adv_img, query_times = attack.generate(all_inputs, (pre_gt_boxes, pre_gt_label))
        adv_imgs.append(adv_img)
        idx += 1
    np.save('ori_imagess.npy', ori_imagess)
    np.save('ori_meta.npy', ori_meta)
    np.save('ori_box.npy', ori_box)
    np.save('ori_labels.npy', ori_labels)
    np.save('ori_gt_num.npy', ori_gt_num)
    np.save('adv_imgs.npy', adv_imgs)
    print('Generate adversarial samples complete.')
