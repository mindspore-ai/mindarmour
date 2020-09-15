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
"""Eval"""
import os
import argparse
import datetime
import mindspore.nn as nn

from mindspore import context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindarmour.utils import LogUtil

from examples.common.networks.vgg.vgg import vgg16
from examples.common.dataset.data_processing import vgg_create_dataset100
from examples.common.networks.vgg.config import cifar_cfg as cfg


class ParameterReduce(nn.Cell):
    """ParameterReduce"""
    def __init__(self):
        super(ParameterReduce, self).__init__()
        self.cast = P.Cast()
        self.reduce = P.AllReduce()

    def construct(self, x):
        one = self.cast(F.scalar_to_array(1.0), mstype.float32)
        out = x*one
        ret = self.reduce(out)
        return ret


def parse_args(cloud_args=None):
    """parse_args"""
    parser = argparse.ArgumentParser('mindspore classification test')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    # dataset related
    parser.add_argument('--data_path', type=str, default='', help='eval data dir')
    parser.add_argument('--per_batch_size', default=32, type=int, help='batch size for per npu')
    # network related
    parser.add_argument('--graph_ckpt', type=int, default=1, help='graph ckpt or feed ckpt')
    parser.add_argument('--pre_trained', default='', type=str, help='fully path of pretrained model to load. '
                        'If it is a direction, it will test all ckpt')

    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='path to save log')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    args_opt = parser.parse_args()
    args_opt = merge_args(args_opt, cloud_args)

    args_opt.image_size = cfg.image_size
    args_opt.num_classes = cfg.num_classes
    args_opt.per_batch_size = cfg.batch_size
    args_opt.momentum = cfg.momentum
    args_opt.weight_decay = cfg.weight_decay
    args_opt.buffer_size = cfg.buffer_size
    args_opt.pad_mode = cfg.pad_mode
    args_opt.padding = cfg.padding
    args_opt.has_bias = cfg.has_bias
    args_opt.batch_norm = cfg.batch_norm
    args_opt.initialize_mode = cfg.initialize_mode
    args_opt.has_dropout = cfg.has_dropout

    args_opt.image_size = list(map(int, args_opt.image_size.split(',')))

    return args_opt


def merge_args(args, cloud_args):
    """merge_args"""
    args_dict = vars(args)
    if isinstance(cloud_args, dict):
        for key in cloud_args.keys():
            val = cloud_args[key]
            if key in args_dict and val:
                arg_type = type(args_dict[key])
                if arg_type is not type(None):
                    val = arg_type(val)
                args_dict[key] = val
    return args


def test(cloud_args=None):
    """test"""
    args = parse_args(cloud_args)
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args.device_target, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    args.logger = LogUtil.get_instance()
    args.logger.set_level(20)

    net = vgg16(num_classes=args.num_classes, args=args)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, args.momentum,
                   weight_decay=args.weight_decay)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    param_dict = load_checkpoint(args.pre_trained)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    dataset_test = vgg_create_dataset100(args.data_path, args.image_size, args.per_batch_size, training=False)
    res = model.eval(dataset_test)
    print("result: ", res)


if __name__ == "__main__":
    test()
