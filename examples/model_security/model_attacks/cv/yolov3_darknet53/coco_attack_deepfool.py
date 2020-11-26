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
"""generate adversarial example for yolov3_darknet53 by DeepFool"""
import os
import argparse
import datetime
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms

from mindarmour.adv_robustness.attacks import DeepFool

from src.yolo import YOLOV3DarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.config import ConfigYOLOV3DarkNet53


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser('mindspore coco testing')

    # device related
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')

    parser.add_argument('--data_dir', type=str, default='', help='train data dir')
    parser.add_argument('--pretrained', default='', type=str, help='model_path, local pretrained model to load')
    parser.add_argument('--samples_num', default=1, type=int, help='Number of sample to be generated.')
    parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')
    parser.add_argument('--testing_shape', type=str, default='', help='shape for test ')

    args, _ = parser.parse_known_args()

    args.data_root = os.path.join(args.data_dir, 'val2014')
    args.annFile = os.path.join(args.data_dir, 'annotations/instances_val2014.json')

    return args


def conver_testing_shape(args):
    """Convert testing shape to list."""
    testing_shape = [int(args.testing_shape), int(args.testing_shape)]
    return testing_shape


class SolveOutput(Cell):
    """Solve output of the target network to adapt DeepFool."""
    def __init__(self, network):
        super(SolveOutput, self).__init__()
        self._network = network
        self._reshape = P.Reshape()

    def construct(self, image, input_shape):
        prediction = self._network(image, input_shape)
        output_big = prediction[0]
        output_big = self._reshape(output_big, (output_big.shape[0], -1, 85))
        output_big_boxes = output_big[:, :, 0: 5]
        output_big_logits = output_big[:, :, 5:]
        return output_big_boxes, output_big_logits


def test():
    """The function of eval."""
    args = parse_args()

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=True, device_id=devid)

    # logger
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(args.outputs_dir, rank_id)

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    args.logger.info('Creating Network....')
    network = SolveOutput(YOLOV3DarkNet53(is_training=False))

    data_root = args.data_root
    ann_file = args.annFile

    args.logger.info(args.pretrained)
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)

    config = ConfigYOLOV3DarkNet53()
    if args.testing_shape:
        config.test_img_shape = conver_testing_shape(args)

    ds, data_size = create_yolo_dataset(data_root, ann_file, is_training=False, batch_size=1,
                                        max_epoch=1, device_num=1, rank=rank_id, shuffle=False,
                                        config=config)

    args.logger.info('testing shape : {}'.format(config.test_img_shape))
    args.logger.info('totol {} images to eval'.format(data_size))

    network.set_train(False)
    # build attacker
    attack = DeepFool(network, num_classes=80, model_type='detection', reserve_ratio=0.9, bounds=(0, 1))
    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)

    args.logger.info('Start inference....')
    batch_num = args.samples_num
    adv_example = []
    for i, data in enumerate(ds.create_dict_iterator(num_epochs=1)):
        if i >= batch_num:
            break
        image = data["image"]
        image_shape = data["image_shape"]

        gt_boxes, gt_logits = network(image, input_shape)
        gt_boxes, gt_logits = gt_boxes.asnumpy(), gt_logits.asnumpy()
        gt_labels = np.argmax(gt_logits, axis=2)

        adv_img = attack.generate((image.asnumpy(), image_shape.asnumpy()), (gt_boxes, gt_labels))
        adv_example.append(adv_img)
    np.save('adv_example.npy', adv_example)


if __name__ == "__main__":
    test()
