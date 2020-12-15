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
"""PGD attack for faster rcnn"""
import os
import argparse
import pickle

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation

from mindarmour.adv_robustness.attacks import ProjectedGradientDescent

from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset

# pylint: disable=locally-disabled, unused-argument, redefined-outer-name

set_seed(1)

parser = argparse.ArgumentParser(description='FasterRCNN attack')
parser.add_argument('--pre_trained', type=str, required=True, help='pre-trained ckpt file path for target model.')
parser.add_argument('--device_id', type=int, default=0, help='Device id, default is 0.')
parser.add_argument('--num', type=int, default=5, help='Number of adversarial examples.')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)


class LossNet(Cell):
    """loss function."""
    def construct(self, x1, x2, x3, x4, x5, x6):
        return x4 + x6


class WithLossCell(Cell):
    """Wrap the network with loss function."""
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_num, *labels):
        loss1, loss2, loss3, loss4, loss5, loss6 = self._backbone(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
        return self._loss_fn(loss1, loss2, loss3, loss4, loss5, loss6)

    @property
    def backbone_network(self):
        return self._backbone


class GradWrapWithLoss(Cell):
    """
    Construct a network to compute the gradient of loss function in \
    input space and weighted by `weight`.
    """
    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = GradOperation(get_all=True, sens_param=False)
        self._network = network

    def construct(self, *inputs):
        gout = self._grad_all(self._network)(*inputs)
        return gout[0]


if __name__ == '__main__':
    prefix = 'FasterRcnn_eval.mindrecord'
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    pre_trained = args.pre_trained

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
                                    repeat_num=1, is_training=True)
    net = Faster_Rcnn_Resnet50(config)
    param_dict = load_checkpoint(pre_trained)
    load_param_into_net(net, param_dict)
    net = net.set_train()

    # build attacker
    with_loss_cell = WithLossCell(net, LossNet())
    grad_with_loss_net = GradWrapWithLoss(with_loss_cell)
    attack = ProjectedGradientDescent(grad_with_loss_net, bounds=None, eps=0.1)

    # generate adversarial samples
    num = args.num
    num_batches = num // config.test_batch_size
    channel = 3
    adv_samples = [0]*(num_batches*config.test_batch_size)
    adv_id = 0
    for data in ds.create_dict_iterator(num_epochs=num_batches):
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        adv_img = attack.generate((img_data.asnumpy(), \
            img_metas.asnumpy(), gt_bboxes.asnumpy(), gt_labels.asnumpy(), gt_num.asnumpy()), gt_labels.asnumpy())
        for item in adv_img:
            adv_samples[adv_id] = item
            adv_id += 1
        if adv_id >= num_batches*config.test_batch_size:
            break

    pickle.dump(adv_samples, open('adv_samples.pkl', 'wb'))
    print('Generate adversarial samples complete.')
