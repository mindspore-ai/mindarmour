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
"""
Training example of suppress-based privacy.
"""
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.nn import Accuracy
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

from mindarmour.privacy.sup_privacy import SuppressModel
from mindarmour.privacy.sup_privacy import SuppressMasker
from mindarmour.privacy.sup_privacy import SuppressPrivacyFactory
from mindarmour.privacy.sup_privacy import MaskLayerDes
from mindarmour.utils import LogUtil

from examples.common.networks.lenet5.lenet5_net import LeNet5
from sup_privacy_config import mnist_cfg as cfg


LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Lenet5_Suppress_train'


def generate_mnist_dataset(data_path, batch_size=32, repeat_size=1, samples=None, num_parallel_workers=1, sparse=True):
    """
    create dataset for training or testing
    """
    # define dataset
    ds1 = ds.MnistDataset(data_path, num_samples=samples)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    if not sparse:
        one_hot_enco = C.OneHot(10)
        ds1 = ds1.map(input_columns="label", operations=one_hot_enco, num_parallel_workers=num_parallel_workers)
        type_cast_op = C.TypeCast(mstype.float32)
    ds1 = ds1.map(input_columns="label", operations=type_cast_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=resize_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=rescale_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=hwc2chw_op,
                  num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    ds1 = ds1.shuffle(buffer_size=buffer_size)
    ds1 = ds1.batch(batch_size, drop_remainder=True)
    ds1 = ds1.repeat(repeat_size)

    return ds1

def mnist_suppress_train(epoch_size=10, start_epoch=3, lr=0.05, samples=10000, mask_times=1000,
                         sparse_thd=0.90, sparse_start=0.0, masklayers=None):
    """
    local train by suppress-based privacy
    """

    networks_l5 = LeNet5()
    suppress_ctrl_instance = SuppressPrivacyFactory().create(networks_l5,
                                                             masklayers,
                                                             policy="local_train",
                                                             end_epoch=epoch_size,
                                                             batch_num=(int)(samples/cfg.batch_size),
                                                             start_epoch=start_epoch,
                                                             mask_times=mask_times,
                                                             lr=lr,
                                                             sparse_end=sparse_thd,
                                                             sparse_start=sparse_start)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.SGD(networks_l5.trainable_params(), lr)
    config_ck = CheckpointConfig(save_checkpoint_steps=(int)(samples/cfg.batch_size),
                                 keep_checkpoint_max=10)

    # Create the SuppressModel model for training.
    model_instance = SuppressModel(network=networks_l5,
                                   loss_fn=net_loss,
                                   optimizer=net_opt,
                                   metrics={"Accuracy": Accuracy()})
    model_instance.link_suppress_ctrl(suppress_ctrl_instance)

    # Create a Masker for Suppress training. The function of the Masker is to
    # enforce suppress operation while training.
    suppress_masker = SuppressMasker(model=model_instance, suppress_ctrl=suppress_ctrl_instance)

    mnist_path = "./MNIST_unzip/"  #"../../MNIST_unzip/"
    ds_train = generate_mnist_dataset(os.path.join(mnist_path, "train"),
                                      batch_size=cfg.batch_size, repeat_size=1, samples=samples)

    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                                 directory="./trained_ckpt_file/",
                                 config=config_ck)

    print("============== Starting SUPP Training ==============")
    model_instance.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), suppress_masker],
                         dataset_sink_mode=False)

    print("============== Starting SUPP Testing ==============")
    ds_eval = generate_mnist_dataset(os.path.join(mnist_path, 'test'),
                                     batch_size=cfg.batch_size)
    acc = model_instance.eval(ds_eval, dataset_sink_mode=False)
    print("============== SUPP Accuracy: %s  ==============", acc)

    suppress_ctrl_instance.print_paras()
if __name__ == "__main__":
    # This configure can run in pynative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.device_target)

    masklayers_lenet5 = []  # determine which layer should be masked

    masklayers_lenet5.append(MaskLayerDes("conv1.weight", 0, False, True, 10))
    masklayers_lenet5.append(MaskLayerDes("conv2.weight", 1, False, True, 150))
    masklayers_lenet5.append(MaskLayerDes("fc1.weight", 2, True, False, -1))
    masklayers_lenet5.append(MaskLayerDes("fc2.weight", 4, True, False, -1))
    masklayers_lenet5.append(MaskLayerDes("fc3.weight", 6, True, False, 50))

    # do suppreess privacy train, with stronger privacy protection and better performance than Differential Privacy
    mnist_suppress_train(10, 3, 0.10, 60000, 1000, 0.95, 0.0, masklayers=masklayers_lenet5)  # used
